Pathway Events
===============

```{eval-rst}

.. code-block:: console

    import scales_nlp

    docket = scales_nlp.Docket.from_ucid('CASE_UCID')
    for entry in docket:
        if entry.event:
            print(entry.row_number, entry.event.name, entry.event.event_type)

```

<br>

Opening Events
---------------

We identify the following opening events on the docket.  Only the first appearance of any of the following events is deemed the case *opening* and will appear as an `entry.event`, however subsequent appearances of these events can be identified using the `entry.labels` object.

**Complaint:** An initial pleading filed by a plaintiff that outlines the claims against a defendant, setting the foundation for a civil lawsuit.

**Inbound Transfer:** An indication that the case has been transferred in from another another court, typically due to a change in jurisdiction or venue.  This can initiate a new docket in the receiving court.

**Indictment:** A formal charge issued by a grand jury, accusing an individual or entity of committing a crime, initiating a criminal prosecution.

**Information:** A formal criminal charge, similar to an indictment, but filed by a prosecutor instead of a grand jury, initiating a criminal case.

**Notice of Removal:** A document filed by a defendant to transfer a case from state court to federal court, typically due to federal jurisdiction or diversity of citizenship between parties.

**Petition:** A formal written request submitted to a court, asking for a specific action or order, often used to initiate certain types of legal proceedings.

<br>

Dispositive Events
---------------

Dispositive events include both events that resolve the case in its entirety and those which dispose of a party or claim.  Because these include partial resolutions, multiple dispositive events can be identified in a single docket.

**Administrative Closing:** A procedural action taken by the court to temporarily remove a case from its active docket, usually pending the resolution of a related matter or awaiting further developments.

**Case Dismissed:** Entries that clearly dismiss the case but do not fall into any of the other dispositive event categories are tagged with this label.

**Consent Decree:** A court-approved agreement between parties to resolve a dispute, often involving the settlement of a lawsuit without an admission of guilt or liability by the defendant.

**Default Judgment:** A ruling in favor of the plaintiff when the defendant fails to respond or appear in court, resulting in an automatic decision without a full trial.

**Outbound Transfer:** An event where a case is transferred from one court to another, typically due to a change in jurisdiction or venue, and is removed from the transferring court's docket.

**Party Resolution:** The settlement of a legal dispute between parties outside of court, often through negotiation or alternative dispute resolution methods, such as mediation or arbitration.

**Remand:** A court order sending a case back to a lower court for further action or reconsideration, often due to procedural errors or new evidence.

**Rule 12b:** A motion to dismiss a case for specific reasons outlined in the Federal Rules of Civil Procedure, such as lack of jurisdiction, improper venue, or failure to state a claim upon which relief can be granted.  Actions on motions to dismiss that partially dismiss the case, including some of the claims or parties, are included.

**Rule 68:** A procedure under the Federal Rules of Civil Procedure that allows a defendant to make a formal settlement offer to the plaintiff, potentially shifting the costs of litigation if the plaintiff ultimately recovers less than the offer amount.

**Summary Judgment:** A court ruling that decides a case without a full trial when there are no genuine disputes over material facts and the moving party is entitled to judgment as a matter of law.  Actions on motions for summary judgment that partially dismiss the case, including some of the claims or parties, are included.

**Trial:** A formal legal proceeding where parties present evidence and arguments to a judge or jury to determine the outcome of a case, either in a criminal prosecution or a civil lawsuit.
