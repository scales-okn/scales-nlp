import pandas as pd
from copy import deepcopy
import scales_nlp

class Docket():
    def __init__(self, ucid, header, entries=None):
        self.ucid = ucid
        self.court = scales_nlp.load_court(ucid.split(";;")[0])
        self.docket_number = ucid.split(";;")[1]
        self.header = header
        self.entries = entries
        self.events = []

        for entry in self.entries:
            entry.docket = self
        
        self.process_events()
        
    def process_events(self):
        for entry in self:
            if 'transfer' in entry.labels:
                for span in entry.spans:
                    if span['entity'] == 'TRANSFER_TO':
                        if span['court'] == 'same':
                            entry.add_label('inbound transfer')
                        elif span['court'] == 'different':
                            entry.add_label('outbound transfer')
                    elif span['entity'] == 'TRANSFER_FROM':
                        if span['court'] == 'same':
                            entry.add_label('outbound transfer')
                        elif span['court'] == 'different':
                            entry.add_label('inbound transfer')

        opening_events = [
            'complaint', 
            'notice of removal', 
            'information', 
            'indictment', 
            'petition',
            'inbound transfer',
            'transfer',
        ]
        openings = []

        for entry in self:
            if not any(x in entry.classifier_labels for x in ['proposed', 'error']):
                for opening_event in opening_events:
                    if opening_event in entry.labels:
                        openings.append(Event(opening_event, event_type='opening', entry=entry))
                        break

        if len(openings) > 0:
            if not all(x.name == 'transfer' for x in openings) or \
                    openings[0].entry.row_number <= min([(len(self) // 2) + 1, 10]):
                opening = openings[0]
                if opening.name in ['transfer', 'inbound transfer']:
                    opening.name = 'inbound transfer'
                    opening.entry.add_label('inbound transfer')
                    opening.entry.remove_label('outbound transfer')
                    self[opening.entry.row_number].event = opening
                self.events.append(opening)
        
        dispositive_events = []
        for entry in self:
            dispositive_event = self.get_dispositive_event(entry)
            if dispositive_event is not None:
                dispositive_events.append(Event(dispositive_event, event_type='dispositive', entry=entry))
        
        added_dispositive_events = []
        for event in reversed(dispositive_events):
            add_event = False
            if event.name not in ['transfer', 'case dismissed']:
                add_event = True
            elif event.name == 'transfer':
                if len(added_dispositive_events) == 0 and (len(self) - event.entry.row_number) < min([(len(self) // 2) - 1, 10]):
                    event.name = 'outbound transfer'
                    event.entry.add_label('outbound transfer')
                    event.entry.remove_label('inbound transfer')
                    self[event.entry.row_number].event = event
                    add_event = True
            elif event.name == 'case dismissed':
                if len(added_dispositive_events) == 0 and all(x.name == 'case dismissed' for x in dispositive_events):
                    add_event = True

            if add_event:
                self.events.append(event)
                added_dispositive_events.append(event.name)
        
        self.events = sorted(self.events, key=lambda x: x.entry.row_number)
        if len(self) == 0:
            self.events = [Event('admin closing', event_type='dispositive', entry=None)]

        for entry in self:
            if entry.event is not None:
                if entry.event not in self.events:
                    entry.event = None

    def get_dispositive_event(self, entry):
        if not any(x in entry.labels for x in ['proposed', 'error']):
            if 'sentence' in entry.labels:
                return 'sentence'
                
            if 'trial' in entry.labels or 'verdict' in entry.labels:
                trial_label = 'trial'
                if 'bench trial' in entry.labels:
                    trial_label = 'bench trial'
                elif 'jury trial' in entry.labels:
                    trial_label = 'jury trial'
                return trial_label
            
            if 'findings of fact' in entry.labels and 'conclusions' in entry.text.lower():
                return 'findings of fact and conclusions'
            
            if 'remand resolution' in entry.labels:
                return 'remand'
            
            if 'default judgment resolution' in entry.labels:
                return 'default judgment'
            
            if 'granting motion for summary judgment' in entry.labels:
                return 'summary judgment'
            
            if 'rule 68 resolution' in entry.labels:
                return 'rule 68'

            if 'consent decree resolution' in entry.labels:
                return 'consent decree'

            if any(x in entry.labels for x in [
                'consent judgment','settlement agreement',
                'motion for settlement', 'notice of settlement', 'stipulation for settlement', 
            ]):
                return 'settlement'

            if 'notice of voluntary dismissal' in entry.labels and 'bilateral' not in entry.labels:
                return 'voluntary dismissal'
            
            if 'bilateral' in entry.labels and any(x in entry.labels for x in [
                'motion for judgment','motion to dismiss',
                'notice of dismissal', 'notice of voluntary dismissal', 'stipulation for judgment', 
                'stipulation of dismissal', 'stipulation for voluntary dismissal',
            ]):
                if any(x in entry.labels for x in ['unopposed', 'stipulation']) and not 'dismiss with prejudice' in entry.labels:
                    return 'voluntary dismissal'
                else:
                    return 'settlement'

            if 'party resolution' in entry.labels:
                return 'party resolution'

            if 'granting motion to dismiss' in entry.labels:
                return 'rule 12b'
            
            if 'outbound transfer' in entry.labels:
                return 'outbound transfer'
            
            if 'transfer' in entry.labels and 'inbound transfer' not in entry.labels:
                return 'transfer'
            
            if 'sentence' in entry.labels:
                return 'sentence'

            if 'case opened in error' in entry.labels:
                return 'admin closing'

            if 'case dismissed' in entry.labels and ' usca ' not in entry.text.lower():
                return 'case dismissed'

    @property
    def opening(self):
        events = [event for event in self.events if event.event_type == 'opening']
        if len(events) > 0:
            return events[0]
        else:
            return Event('unknown', event_type='opening')
    
    @property
    def dispositive_events(self):
        return [event for event in self.events if event.event_type == 'dispositive']

    @property
    def nos(self):
        nos_code = self.header['nature_suit']
        if nos_code is not None:
            nos_code = nos_code.split()[0]
            if nos_code.isdigit():
                return int(nos_code)
        
    @staticmethod
    def from_json(case_json, label_json=None, recap=False):
        if not recap:
            entries = []
            if label_json is not None:
                label_json = {x['row_number']: x for x in label_json}
            for row_number, entry in enumerate(case_json['docket']):
                entry['entry_number'] = entry['ind']
                entry['text'] = entry['docket_text']
                del entry['ind']
                del entry['docket_text']
                classifier_labels = None if label_json is None else label_json.get(row_number, {}).get('labels', [])
                classifier_spans = None if label_json is None else label_json.get(row_number, {}).get('spans', [])
                entries.append(DocketEntry(
                    row_number=row_number,
                    classifier_labels=classifier_labels,
                    classifier_spans=classifier_spans,
                    **entry
                ))

            del case_json['docket']
            return Docket(
                ucid=case_json['ucid'],
                header=case_json,
                entries=entries,
            )
    
    @staticmethod
    def from_ucid(ucid):
        case_json = scales_nlp.load_case(ucid)
        label_json = scales_nlp.load_case_classifier_labels(ucid)
        return Docket.from_json(case_json, label_json=label_json)

    def __iter__(self):
        for entry in sorted(self.entries, key=lambda x: x.row_number):
            yield entry
    
    def __getitem__(self, *args):
        return self.entries.__getitem__(*args)
    
    def __len__(self):
        return len(self.entries)
    
    def __repr__(self):
        return f"<Docket: {self.ucid}>"


class DocketEntry():
    def __init__(
        self, row_number, entry_number=None, date_filed=None, text=None, 
        documents=None, edges=None, classifier_labels=None, classifier_spans=None, docket=None,
    ):
        self.row_number = row_number
        self.entry_number = entry_number
        self.date_filed = pd.to_datetime(date_filed, errors='coerce')
        self.text = text
        self.documents = documents
        self.edges = edges
        self.classifier_labels = classifier_labels
        self.classifier_spans = classifier_spans
        self._labels = None
        self._spans = None
        self.event = None
        self.docket = docket
    
    def add_label(self, label):
        self._labels = self.labels + [label]
        self._labels = self.get_labels(update=True)


    def remove_label(self, label):
        self._labels = [x for x in self.labels if x != label]
        self._labels = self.get_labels(update=True)

    def to_json(self):
        return {
            'ord': self.row_number,
            'entry_number': self.entry_number,
            'date_filed': self.date_filed,
            'text': self.text,
            'documents': self.documents,
            'edges': self.edges,
            'classifier_labels': self.classifier_labels,
            'classifier_spans': self.classifier_spans,
            'labels': self.labels,
        }
    
    def __repr__(self):
        if self.docket is not None:
            return f"<DocketEntry: {self.docket.ucid} [{self.row_number}]>"
        else:
            return f"<DocketEntry: unknown_docket [{self.row_number}]>"

    def get_labels(self, update=False):
        if self._labels is None:
            self._labels = deepcopy(self.classifier_labels)
            update = True
        if update:
            self._labels = list(sorted(list(set(self._labels))))
        return self._labels
    
    def get_spans(self, update=False):
        if self._spans is None:
            self._spans = deepcopy(self.classifier_spans)
            update = True

        if update:
            spans = []
            for span in self._spans:
                if span['entity'] in ['TRANSFER_TO', 'TRANSFER_FROM']:
                    courts = scales_nlp.courts()
                    states = [x for x in scales_nlp.states() if x.lower() in span['text'].lower()]
                    divisions = [x.lower() for x in scales_nlp.divisions() if x.lower() in span['text'].lower()]
                    if len(states) > 0:
                        courts = courts[courts['state'].isin(states)]
                    if len(divisions) > 0:
                        courts = courts[courts['cardinal'].str.lower().isin(divisions)]
                    possible_matches = courts['abbreviation'].unique()
                    if self.docket.court['abbreviation'] not in possible_matches:
                        span['court'] = 'different'
                    elif len(possible_matches) == 1 and possible_matches[0] == self.docket.court['abbreviation']:
                        span['court'] = 'same'
                    else:
                        span['court'] = 'unknown'
                spans.append(span)
            self._spans = list(sorted(sorted(spans, key=lambda x: x['entity']), key=lambda x: x['start']))
        return self._spans
    
    @property
    def labels(self):
        return self.get_labels()

    @property
    def spans(self):
        return self.get_spans()
    

class Event():
    def __init__(self, name, event_type, entry=None):
        self.name = name
        self.event_type = event_type
        self.entry = entry
        if entry is not None:
            entry.event = self
    
    def __repr__(self):
        event_str = '' if self.entry is None else f" [{self.entry.row_number}]"
        return f"<Event: {self.name} ({self.event_type}){event_str}>"

