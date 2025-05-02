import re
import json
import pandas as pd
from copy import deepcopy
from itertools import chain
from fuzzywuzzy import fuzz
import scales_nlp

label_remappings = {
    'admin closing': ('attribute_admin_closing',),
    'arbitration motion': ('motion', 'attribute_motion_for_arbitration'),
    'bench trial': ('trial', 'attribute_trial_bench'),
    'case dismissed': ('attribute_dismissal_other',),
    'case opened in error': ('attribute_case_opened_in_error',),
    'default judgment': ('attribute_default_judgment',),
    'default judgment resolution': ('attribute_default_judgment',),
    'dismiss with prejudice': ('attribute_dismiss_with_prejudice',),
    'dismiss without prejudice': ('attribute_dismiss_without_prejudice',),
    'dismissing motion': ('motion', 'attribute_motion_for_dismissal_other'),
    'error': ('attribute_error',),
    'findings of fact': ('findings_of_fact', 'attribute_dispositive'),
    'granting motion for summary judgment': ('attribute_granting_motion_for_summary_judgment', 'attribute_dispositive'),
    'granting motion to dismiss': ('attribute_granting_motion_to_dismiss', 'attribute_dispositive'),
    'guilty plea': ('plea', 'attribute_plea_guilty'),
    'inbound transfer': ('attribute_transfer_inbound',),
    'jury trial': ('trial', 'attribute_trial_jury'),
    'not guilty plea': ('plea', 'attribute_plea_not_guilty'),
    'notice of removal': ('removal',),
    'other trial': ('trial', 'attribute_trial_other'),
    'outbound transfer': ('attribute_transfer_outbound',),
    'proposed': ('attribute_proposed',),
    'remand': ('attribute_remand',),
    'rule 68': ('settlement', 'attribute_settlement_rule_68'),
    'transfer': ('attribute_transfer_unknown',),
    'transferred entry': ('attribute_transferred_entry',),
    'trial': ('trial', 'attribute_trial_other'),
    'verdict': ('verdict', 'attribute_dispositive'),
    'voluntary dismissal': ('attribute_voluntary_dismissal',),
    'voluntary dismissal (settlement)': ('settlement', 'attribute_dispositive')
}
labels_to_delete = (
    'agreement reached',
    'consent decree resolution',
    'consent judgment',
    'consent judgment resolution',
    'remand resolution',
    'rule 12b',
    'rule 68 resolution',
    'settlement agreement',
    'settlement reached',
    'summary judgment',
    'voluntary dismissal resolution'
)
labels_to_change_to_order = (
    'admin closing',
    'case dismissed',
    'default judgment',
    'granting motion for summary judgment',
    'granting motion to dismiss',
    'inbound transfer',
    'outbound transfer',
    'remand'
)
labels_auto_dispositive = (
    'attribute_admin_closing',
    'attribute_default_judgment',
    'findings_of_fact',
    'attribute_granting_motion_for_summary_judgment',
    'attribute_granting_motion_to_dismiss',
    'attribute_transfer_outbound',
    'attribute_remand',
    'sentence',
    'settlement',
    'verdict',
    'attribute_voluntary_dismissal'
)
labels_to_change_to_settlement = (
    'agreement reached',
    'consent decree',
    'consent judgment',
    'notice of consent',
    'notice of dismissal',
    'notice of settlement',
    'notice of voluntary dismissal',
    'settlement agreement',
    'settlement reached',
    'stipulation for judgment',
    'stipulation for settlement',
    'stipulation of dismissal'
)
deceptive_nonmotion_labels = (
    'granting motion for summary judgment',
    'granting motion to dismiss',
    'notice of motion'
)
event_labels = (
    'answer',
    'arrest',
    'brief',
    'complaint',
    'findings_of_fact',
    'indictment',
    'information',
    'judgment',
    'minute_entry',
    'motion',
    'notice',
    'order',
    'petition',
    'plea',
    'plea_agreement',
    'removal',
    'response',
    'sentence',
    'settlement',
    'stipulation',
    'summons',
    'trial',
    'verdict',
    'waiver',
    'warrant'
)

class Docket():
    def __init__(self, ucid, header, entries=None, judge_df=None, skip_monkey_patch=False):
        self.ucid = ucid
        self.court = scales_nlp.load_court(ucid.split(";;")[0])
        self.docket_number = ucid.split(";;")[1]
        self.header = header
        self.entries = entries
        self.judge_df = judge_df
        self.events = []

        for entry in self.entries:
            entry.docket = self
        
        self.process_events(skip_monkey_patch)
        
    def process_events(self, skip_monkey_patch):
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
            if event.name == 'transfer':
                if len(added_dispositive_events) == 0 and (len(self) - event.entry.row_number) < min([(len(self) // 2) - 1, 10]):
                    event.name = 'outbound transfer'
                    event.entry.add_label('outbound transfer')
                    event.entry.remove_label('inbound transfer')
                    self[event.entry.row_number].event = event
                    add_event = True
            elif event.name == 'case dismissed':
                if len(added_dispositive_events) == 0 and all(x.name == 'case dismissed' for x in dispositive_events):
                    add_event = True
            elif event.name == 'admin closing':
                if len(added_dispositive_events) == 0 and all(x.name in ['case dismissed', 'admin closing'] for x in dispositive_events):
                    add_event = True
            else:
                add_event = True

            if add_event:
                self.events.append(event)
                added_dispositive_events.append(event.name)
        
        self.events = sorted(self.events, key=lambda x: x.entry.row_number)
        if len(self) == 0:
            self.events = [Event('admin closing', event_type='dispositive', entry=None)]

        if skip_monkey_patch:
            for entry in self:
                if entry.event is not None:
                    if entry.event not in self.events:
                        entry.event = None
            return

        # monkey patch by scott, starting with event handling
        for entry in self:
            if entry.event is not None:
                if entry.event in self.events and entry.event.name not in labels_to_delete:
                    if entry.event.event_type == 'opening':
                        entry.add_label_basic('attribute_opening')
                    if entry.event.event_type == 'dispositive' and 'trial' not in entry.event.name:
                        entry.add_label_basic('attribute_dispositive')
                    entry.add_label_basic(entry.event.name)
                entry.event = None

            # pre-remapping changes
            labels_old = deepcopy(entry.labels)
            labels_and_remappings = labels_old + list(filter(None, chain.from_iterable([label_remappings.get(x) or (None,) for x in labels_old])))
            for label in labels_old:
                if label in labels_to_change_to_order and 'minute entry' not in labels_old:
                    entry.add_label_basic('order')
                    if label not in label_remappings:
                        entry.remove_label_basic(label)
                delete_mode = False
                if label in labels_to_change_to_settlement:
                    if 'case dismissed' in labels_old or 'settlement' in labels_and_remappings or label=='consent decree':
                        entry.add_label_basic('settlement')
                        delete_mode = True
                    if label=='consent decree':
                        entry.add_label_basic('attribute_settlement_consent_decree')
                    if delete_mode:
                        entry.remove_label_basic(label)
                        continue

                # label remapping
                if label in label_remappings:
                    entry.remove_label_basic(label)
                    for label_new in label_remappings[label]:
                        entry.add_label_basic(label_new)
                elif label in labels_to_delete:
                    entry.remove_label_basic(label)
                elif any(x in label for x in ('motion', 'notice', 'petition', 'stipulation', 'waiver')):
                    if ' ' not in label:
                        if len([x for x in labels_old if label in x and not (label=='motion' and x in deceptive_nonmotion_labels)])==1:
                            entry.add_label_basic(f'attribute_{label}_other')
                    else:
                        entry.add_label_basic(label.split(' ')[0])
                        entry.remove_label_basic(label)
                        entry.add_label_basic(f"attribute_{label.replace(' ','_') + ('_other' if label in ('motion for judgment', 'notice of dismissal') else '')}")
                elif label in ('bilateral', 'unopposed'):
                    entry.remove_label_basic(label)
                    if not any(('settlement' in x or 'stipulation' in x) for x in labels_old):
                        entry.add_label_basic('attribute_bilateral_unopposed')
                elif ' ' in label:
                    entry.remove_label_basic(label)
                    entry.add_label_basic(label.replace(' ','_'))

            # various post-remapping changes
            if 'settlement' in entry.labels:
                labels_to_delete_due_to_settlement = [x for x in entry.labels if 'stipulation' in x or x=='attribute_voluntary_dismissal']
                for l in labels_to_delete_due_to_settlement:
                    entry.remove_label_basic(l)
            if 'settlement' in entry.labels or any('stipulation' in x for x in entry.labels):
                entry.remove_label_basic('attribute_bilateral_unopposed')
            if 'attribute_voluntary_dismissal' in entry.labels and 'attribute_granting_motion_to_dismiss' in entry.labels:
                entry.remove_label_basic('attribute_granting_motion_to_dismiss')
            if any(x in entry.labels for x in labels_auto_dispositive):
                entry.add_label_basic('attribute_dispositive')
                if 'attribute_dismissal_other' in entry.labels:
                    entry.remove_label_basic('attribute_dismissal_other')
            for label_intermediate, labels_specific in {
                'attribute_motion_for_dismissal_other': ('attribute_motion_for_voluntary_dismissal', 'attribute_motion_to_dismiss'),
                'attribute_motion_for_judgment_other': ('attribute_motion_for_default_judgment', 'attribute_motion_for_judgment_as_a_matter_of_law',
                    'attribute_motion_for_judgment_on_the_pleadings', 'attribute_motion_for_summary_judgment'),
                'attribute_notice_of_dismissal_other': ('attribute_notice_of_voluntary_dismissal',)
            }.items():
                if any(x in entry.labels for x in labels_specific):
                    entry.remove_label_basic(label_intermediate)
            for keyword,label_to_remove in (('attribute_transfer','attribute_transfer_unknown'), ('attribute_trial','attribute_trial_other')):
                if len([x for x in entry.labels if keyword in x])>1:
                    entry.remove_label_basic(label_to_remove)
            if 'attribute_proposed' in entry.labels:
                entry.remove_label_basic('attribute_dispositive')

            # post-remapping judge-action logic
            jdg_act_type = entry.detect_judge_action()
            events_of_interest = [x for x in event_labels if x in entry.labels and x not in (
                'order','settlement','plea')] # i believe orders and settlements can coexist (see comment on 'consent judgment resolution' in get_dispositive_event)
            if 'order' in entry.labels and events_of_interest:
                entry.remove_label_basic('attribute_proposed')
                if jdg_act_type: # accept "strong" or "weak", decline None
                    for event in events_of_interest:
                        for attr in [x for x in entry.labels if event in x]:
                            entry.remove_label_basic(attr)
                else:
                    entry.remove_label_basic('order')

            # other post-remapping multiple-event corrections
            if 'petition' in entry.labels and 'response' in entry.labels and 'petition' not in entry.text.lower().replace("petitioner", '').replace(
                'response to petition', '') or not entry.text.lower().split("petitioner's response")[0].split("petitioner's reply")[0]:
                entry.remove_label_basic('petition')
            if 'order' in entry.labels and 'answer' in entry.labels and 'order' not in entry.text.lower().replace('scheduling order due', ''):
                entry.remove_label_basic('order')
            problematic_settlement_text = 'due to civil action being compromised and settled'
            if 'settlement' in entry.labels and problematic_settlement_text in entry.text.lower() and 'settl' not in entry.text.lower().replace(problematic_settlement_text, ''):
                entry.remove_label_basic('settlement')


    def get_dispositive_event(self, entry):
        if not any(x in entry.labels for x in ['proposed', 'error']):
            if 'sentence' in entry.labels:
                return 'sentence'
                
            ffc = 'findings of fact' in entry.labels and 'conclusions' in entry.text.lower()
            if 'trial' in entry.labels: # or 'verdict' in entry.labels or ffc:
                trial_label = 'other trial'
                if 'bench trial' in entry.labels:
                    trial_label = 'bench trial'
                elif 'jury trial' in entry.labels:
                    trial_label = 'jury trial'
                return trial_label
                
            if 'remand resolution' in entry.labels:
                return 'remand'
            
            if 'default judgment resolution' in entry.labels:
                return 'default judgment'
            
            if 'granting motion for summary judgment' in entry.labels:
                return 'summary judgment'
            
            if 'rule 68 resolution' in entry.labels:
                return 'rule 68'

            if 'consent decree resolution' in entry.labels:
                # entries that are consent decrees
                # orders granting motions for consent decrees
                # orders disposing of the case via consent decree
                return 'consent decree'

            is_vd, is_settlement = False, False

            if any(x in entry.labels for x in [
                'voluntary dismissal resolution', 
                # captures notices of dismissal, stipulations of dismissal
                # orders granting on the basis of notices / stipulation of dismissal
                # orders disposing of cases via 'voluntary dismissal' or rule 41(a)
                'stipulation of dismissal', # entries that are stipulations of dismissal (redundant)
                'notice of dismissal', # entries that are notices of dismissal (redundant)
                'notice of voluntary dismissal', # entries that are notices of voluntary dismissal (redundant)
            ]):
                is_vd = True

            if is_vd and 'dismissed with prejudice' in entry.labels:
                is_settlement = True

            if any(x in entry.labels for x in [
                'settlement reached', # catch all label for any indicator that the parties settled using the language of settlement
                'consent judgment', # entries that are consent judgments
                'consent judgment resolution', # orders granting motions for or disposing of case via consent judgments
                'settlement agreement', # entries that are settlement agreements
                'motion for settlement', # entries that are motions for settlement
                'notice of settlement', # entries that are notices of settlement
                'stipulation for settlement', # entries that are stipulations for settlement
            ]):
                is_settlement = True

            if is_vd and is_settlement:
                return 'voluntary dismissal (settlement)'
            elif is_vd:
                return 'voluntary dismissal'
            elif is_settlement:
                return 'settlement'

            # what to do with motion / stipulation for judgment (prev included if bilateral)
   
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


    def get_party_names(self, party_type=None):
        names = []
        for party in self.header['parties']:
            if party_type is None or party['party_type'] == party_type:
                names.append(party['name'])
        return names

    @property
    def plaintiff_names(self):
        return self.get_party_names(party_type='plaintiff')
    
    @property
    def defendant_names(self):
        return self.get_party_names(party_type='defendant')

    def get_attorney_names(self, party_type=None):
        names = []
        for party in self.header['parties']:
            if party_type is None or party['party_type'] == party_type:
                for attorney in party['counsel']:
                    names.append(attorney['name'])
        return names

    @property
    def plaintiff_attorney_names(self):
        return self.get_attorney_names(party_type='plaintiff')
    
    @property
    def defendant_attorney_names(self):
        return self.get_attorney_names(party_type='defendant')

    @staticmethod
    def from_json(case_json, label_json=None, judge_df=None, recap=False, skip_monkey_patch=False):
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
                judge_df=judge_df,
                skip_monkey_patch=skip_monkey_patch
            )
    
    @staticmethod
    def from_ucid(ucid, skip_monkey_patch=False):
        label_json = scales_nlp.load_case_classifier_labels(ucid)
        case_json = scales_nlp.load_case(ucid)
        judge_df = scales_nlp.load_case_judge_labels(ucid)
        return Docket.from_json(case_json, label_json=label_json, judge_df=judge_df, skip_monkey_patch=skip_monkey_patch)

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
        documents=[], edges=[], classifier_labels=None, classifier_spans=None, docket=None,
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
    
    def add_label(self, label, update=True, no_dups=False):
        self._labels = list(set(self.labels + [label])) if no_dups else self.labels + [label]
        self._labels = self.get_labels(update=update)

    def remove_label(self, label, update=True):
        self._labels = [x for x in self.labels if x != label]
        self._labels = self.get_labels(update=update)

    # not sure why update defaults to True, but I didn't want to change it and didn't want to pass update=False every time, so I wrote the below two functions

    def add_label_basic(self, label):
        self.add_label(label, update=False, no_dups=True)

    def remove_label_basic(self, label):
        self.remove_label(label, update=False)

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

    def detect_judge_action(self):
        '''
        Determines wheter a judge took action in the course of the event described by the given docket entry.
        output:
            - one of 'strong' (a judge took action),
                     'weak' (a judge was mentioned in a court-initiated action), or
                     None (no judge was mentioned OR the action was party-initiated)
        '''

        # set up variables/constants/helpers         
        jdata = None
        is_strong, is_weak, is_potentially_party_initiated = False, False, False
        date_re = r'\d{1,2}/\d{1,2}/\d{2,4}'
        blanket_cases_strong_re = ''.join((fr'(?i)(?:electronic |paperless )?',
            r'(?:minute (?:entry ?(?:for proceedings (?:held )?)?(?:(?:on )?{date_re} )?before|',
            r'order (?:in chambers of|issued by))|',
            r"(?:clerk's )?(?:minutes|notes) (?:for|of) [a-z/ ]+ before|",
            r'(?:magistrate )?judge [a-z\., ]+: (?:electronic )?order entered)'))
        blanket_cases_weak_re = ''.join(('(?i)(?:electronic )?(?:(?:initial|notice of) )?(?:(?:case|judge) )?(?:re)?assign(?:ed|ment)|',
            r"(?:text only entry: )?clerk'?s notice of (?:(?:initial case|(?:magistrate )?judge) assignment|reassignment)|",
            r'(?:magistrate )?(?:judge|hon\.) [a-z\., ]+ (?:added|assigned to case|is so designated)\.|',
            r'case (?:referred to|opening (?:initial assignment notice|notification))|',
            r'this case has been assigned|random assignment of magistrate judge|',
            r'order (?:reassigning case|that this case is reassigned)|',
            r'civil case terminated\. magistrate judge [a-z\., ]+ terminated from case\.|',
            r'this action has been transferred|',
            r'action required by (?:district|magistrate) judge|',
            r'new case notes'))
        party_keywords = ['plaintiff', 'plaintiffs', 'defendant', 'defendants', 'usa', 'united states', 'united states of america']
        _clean_word = lambda x: x.lower().replace('.','').strip('(')
        def _is_scheduling_entry(docket_text, span, w2):
            if (re.match(date_re, w2) or w2 in ('am', 'pm', 'courtroom', 'chambers', 'telephone', 'tower)') or re.match(
                r'(?i)\d*\-?[a-z]$', w2) or w2.strip('),').isnumeric() or not docket_text.lower().split(
                'notice of motion')[0]):
                return True
            else:
                return False  

        # load the relevant entry data (this is a translation to the variable names as originally written in data_tools/scales_nlp)
        ucid = self.docket.ucid
        scales_ind = self.row_number
        docket_text = self.text
        judge_df = self.docket.judge_df

        # load the SEL data
        if not len(judge_df):
            return None
        subdf = judge_df[judge_df.docket_index.eq(scales_ind)] # could be optimized if this function ends up running over entire cases
        if not len(subdf):
            return None
        spans = [(subdf.at[i,'Ent_span_start'], subdf.at[i,'Ent_span_end']) for i in subdf.index]

        # for each judge span, take note of the two words preceding it
        preceding_words = []
        _clean_word = lambda x: x.lower().replace('.','').strip('(')
        for span in spans:
            words = docket_text[:span[0]].split()
            i = len(words)-1
            while i>=0 and _clean_word(words[i]) in ('the', 'judge', 'judge:', 'magistrate', 'chief', 'district', 'honorable', 'hon', 'senior', 'united', 'states', 'us'):
                i -= 1
            if i>=0:
                preceding_words.append((_clean_word(words[i]), (_clean_word(words[i-1]) if i>0 else '')))
            else:
                preceding_words.append(('', ''))

        # apply blanket heuristics
        if re.match(blanket_cases_strong_re, docket_text):
            is_strong = True
        elif re.match(blanket_cases_weak_re, docket_text):
            is_weak = True

        # apply per-span heuristics
        if not is_strong:
            for i,span in enumerate(spans):
                w1,w2 = preceding_words[i]
                if any((
                    w1=='by',
                    (w2,w1)==('order','from'),
                    docket_text.count('.')>1 and any((re.match(fr'(?i) (?:magistrate )?judge [a-z\. ]+ on {date_re}', x) for x in (
                        docket_text.split('.')[-2], '.'.join((docket_text.split('.')[-3], docket_text.split('.')[-2]))))))): # wow, sorry for the seven closing parentheses
                    is_strong = True
                    break
                elif not is_weak and any((
                    w1=='and', (docket_text[span[1]+1:].split() and docket_text[span[1]+1:].split()[0]=='and'), # connotes judge-assignment activity ("judge X and judge Y")
                    (w2,w1)==('calendar','of'),
                    w1=='before' and _is_scheduling_entry(docket_text, span, w2))):
                    is_weak = True
                elif any((
                    w1=='to',
                    re.match('(?i)complaint', docket_text))):
                    is_potentially_party_initiated = True

        # apply final catch-all case
        if not is_strong and not is_weak:
            parties = [(x['name'] or '') for x in self.docket.header['parties']]
            first_sentence = (re.match(r'.*?[^\. ]{2}\.', docket_text.replace('..','.')) or re.match('.*', docket_text)).group(0)
            if not any(f'by {x.lower()}' in (first_sentence or '').lower() for x in parties+party_keywords) and not is_potentially_party_initiated:
                is_weak = True

        # finish up
        if is_strong:
            return 'strong'
        elif is_weak:
            return 'weak'
        else:
            return None

    def get_labels(self, update=False):
        if self._labels is None:
            self._labels = deepcopy(self.classifier_labels)
            update = True
        if update:
            text = self.text.lower()

            # add settlement label if bilateral dismissal entry
            if any(x in self._labels for x in ['judgment', 'order', 'case dismissed']):
                first_words = ' '.join(text.split()[:7])
                if 'dismiss' in first_words:
                    if any([x in first_words for x in ['agree', 'consent']]):
                        self._labels.append('settlement reached')

            # jury / bench trial keyword conditions
            if 'trial' in self._labels:
                if 'jury trial' in text and not any(y in text for y in ['non jury trial', 'non-jury trial']):
                    self._labels.append('jury trial')
                if 'bench trial' in text:
                    self._labels.append('bench trial')
            if '[transferred from' in text:
                self._labels.append('transferred entry')

            # override consent decree and consent judgment labels if related to forfeiture or forclosure
            if any(x in text for x in [' forfeit', ' forclos']):
                for x in ['consent decree', 'consent decree resolution', 'consent judgment', 'consent judgment resolution']:
                    if x in self._labels:
                        self._labels.remove(x)
            
            # only include plea if neither a not guilty arraignment or a judgment
            # add not guilty and guilty variations of plea label
            if 'plea' in self._labels:
                if 'judgment' in self._labels:
                    self._labels.remove('plea')
                else:
                    plea_text = text.replace('rearraign', '').replace('re-arraign', '').replace('re arraign', '')
                    if 'arraign' in plea_text and 'not guilty' in plea_text:
                        self._labels.remove('plea')
                    else:
                        if 'not guilty' in plea_text:
                            self._labels.append('not guilty plea')
                            
                        plea_text = plea_text.replace('not guilty', '')
                        if 'guilty' in plea_text:
                            self._labels.append('guilty plea')
                    

            # change motion type from 12b to 41a if strong 41 language in motion
            if 'dismissing motion' in self._labels:
                if any(x in text for x in ['voluntar', '41(a)', '41a']):
                    self._labels.append('motion for voluntary dismissal')
                    if 'motion to dismiss' in self._labels:
                        self._labels.remove('motion to dismiss')

            # change related motion type from 12b to 41a if strong 41 language in order
            if 'voluntary dismissal resolution' in self._labels:
                for span in self.spans:
                    if span['entity'] == 'GRANT' and 'related_entry' in span:
                        related_entry = self.docket[span['related_entry']]
                        if 'dismissing motion' in related_entry.labels:
                            related_entry.add_label('motion for voluntary dismissal')
                            if 'motion to dismiss' in related_entry.labels:
                                related_entry.remove_label('motion to dismiss')

            # change order type from 12b to 41a if strong 41 language in related motion
            if 'granting motion to dismiss' in self._labels and 'voluntary dismissal resolution' not in self._labels:
                for span in self.spans:
                    if span['entity'] == 'GRANT' and 'related_entry' in span:
                        related_entry = self.docket[span['related_entry']]
                        if any(x in related_entry.labels for x in ['motion for voluntary dismissal', 'stipulation of dismissal', 'notice of dismissal']):
                            self._labels.append('voluntary dismissal resolution')
            
            # if order not VD, is granting MTD, and does not have strong 12b language, then if related motion is plaintiff or multi-filed cahnge to 41a
            if 'granting motion to dismiss' in self._labels and 'voluntary dismissal resolution' not in self._labels:
                mtd_terms = [
                    '12b', '12(b)', 'failure to state a claim', 'service of process', 'insufficiency of process', 'insufficient process',
                    'personal jurisdiction', 'subject matter jurisdiction'
                ]
                if not any(x in text for x in mtd_terms):
                    for span in self.spans:
                        if span['entity'] == 'GRANT' and 'related_entry' in span:
                            related_entry = self.docket[span['related_entry']]
                            if 'dismissing motion' in related_entry.labels:
                                if not any(x in related_entry.text.lower() for x in mtd_terms):
                                    if related_entry.filed_by in ['plaintiff', 'multi']:
                                        related_entry.add_label('motion for voluntary dismissal')
                                        if 'motion to dismiss' in related_entry.labels:
                                            related_entry.remove_label('motion to dismiss')
                                        self._labels.append('voluntary dismissal resolution')

            self._labels = list(sorted(list(set(self._labels))))
        return self._labels
    
    def get_spans(self, update=False):
        if self._spans is None:
            self._spans = deepcopy(self.classifier_spans)
            update = True

        if update:
            spans = []
            for span in self._spans:
                # fix this in the models
                span['entity'] = span['entity'].upper().replace(' ', '_')
                if span['entity'] in ['TRANSFER_TO', 'TRANSFER_FROM']:
                    if self.docket.court is None:
                        span['court'] = 'unknown'
                    else:
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
                elif span['entity'] == 'ENTERED_BY':
                    for name in self.docket.plaintiff_names + self.docket.plaintiff_attorney_names:
                        if fuzz.token_set_ratio(span['text'], name) > 80:
                            span['party_type'] = 'plaintiff'
                            span['party'] = name
                            break
                    for name in self.docket.defendant_names + self.docket.defendant_attorney_names:
                        if fuzz.token_set_ratio(span['text'], name) > 80:
                            span['party_type'] = 'defendant'
                            span['party'] = name
                            break
                elif span['entity'] in ['GRANT', 'DENY', 'MOOT', 'PARTIAL']:
                    for edge in self.edges:
                        if edge[-1]['end'] == span['end']:
                            span['related_entry'] = edge[1]
                            break
                    if 'related_entry' not in span:
                        if span['text'].isdigit():
                            for entry in self.docket:
                                if entry.entry_number == int(span['text']):
                                    span['related_entry'] = entry.row_number
                                    break
                spans.append(span)
            self._spans = list(sorted(sorted(spans, key=lambda x: x['entity']), key=lambda x: x['start']))
        return self._spans
    
    @property
    def labels(self):
        return self.get_labels()

    @property
    def spans(self):
        return self.get_spans()
    
    @property
    def filed_by(self):
        filing_party = {'defendant': 0, 'plaintiff': 0}
        for span in self.spans:
            if 'party_type' in span:
                if span['party_type'] in filing_party:
                    filing_party[span['party_type']] += 1
        if filing_party['defendant'] + filing_party['plaintiff'] == 0:
            return None
        elif filing_party['defendant'] == filing_party['plaintiff']:
            return 'multi'
        elif filing_party['defendant'] > filing_party['plaintiff']:
            return 'defendant'
        elif filing_party['defendant'] < filing_party['plaintiff']:
            return 'plaintiff'
    

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

