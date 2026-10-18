"""The 20 neutral sweep prompts for Stage 3.

Neutral = not priming either injected cluster (no arachnid/limb/counting
content, no music/instrument content). Each carries its own ordinary content
words, which become the incumbent tokens the injection has to compete with.
"""

NEUTRAL_PROMPTS = [
    "The harbor was crowded with fishing boats waiting for the tide to turn.",
    "She filed the quarterly report just before the accountant left the office.",
    "Rain collected in the gutter and ran down toward the storm drain.",
    "The bakery on the corner sells bread that sells out by early morning.",
    "He repaired the tractor engine using a wrench borrowed from his neighbor.",
    "The library basement holds newspapers printed during the previous century.",
    "A thin layer of frost covered the windshield of the parked car.",
    "The committee postponed the vote until the budget figures were verified.",
    "Salt water corroded the iron railing along the pier over many winters.",
    "The surgeon reviewed the chart before scrubbing in for the procedure.",
    "Wheat fields stretched to the horizon under a flat grey sky.",
    "The printer jammed again, so the clerk opened the paper tray.",
    "Volcanic soil makes the valley unusually fertile for growing coffee.",
    "The lawyer requested an extension because the witness was traveling abroad.",
    "A copper kettle sat on the stove, steaming gently in the kitchen.",
    "The train slowed as it approached the tunnel beneath the river.",
    "Researchers measured the glacier retreat using satellite photographs.",
    "The tailor marked the fabric with chalk before cutting the sleeve.",
    "Dust gathered on the shelves of the abandoned schoolhouse classroom.",
    "The referee stopped play after the defender slipped on the wet turf.",
]

STOPWORDS = set("""
a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its he she they them his her their
we you i not no so such into over under out up down about after before during
until while because just again only very own same too can will would could
should may might must has have had do does did done there here when where
""".split())
