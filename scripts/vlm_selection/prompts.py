"""
prompts.py
Promptul de analiză a scenei și schema JSON de ieșire.

Sursa de adevăr este `05_recommend.py` (SCENE_ANALYSIS_PROMPT). Aici este replicat
și version-pinned pentru reproducibilitatea studiului. Orice modificare a promptului
trebuie să incrementeze PROMPT_VERSION.
"""

PROMPT_VERSION = "v1.0"

# Identic cu SCENE_ANALYSIS_PROMPT din 05_recommend.py (version-pinned).
SCENE_ANALYSIS_PROMPT = """Analyze this room scene. A user has clicked on a furniture item at approximately the marked position.

Respond in JSON format only:
{
  "room_style": "<one of: Modern, Minimalist, Nordic, Industrial, Art Deco, European Traditional, Southeast Asian, Chinese Traditional, Japanese, Luxury, Kids, Mixed>",
  "selected_object": "<brief description of the clicked object, e.g. 'a beige fabric sofa'>",
  "selected_category": "<one of: Sofa, Chair, Table, Bed, Cabinet, Nightstand, Shelf, Desk, TV Stand, Floor Lamp, Ceiling Lamp, Pendant Lamp, Wardrobe, Bookcase>",
  "replacement_query": "<a 2-sentence description of the IDEAL replacement furniture piece that would fit this room's style and the user's selection>",
  "reasoning": "<why this style and category were inferred>"
}"""

# Câmpurile obligatorii ale schemei (pentru conformitate — D1).
REQUIRED_KEYS = [
    "room_style",
    "selected_object",
    "selected_category",
    "replacement_query",
    "reasoning",
]

# Sarcina la nivel de OBIECT (substrat 3D-FUTURE, complet automat, fără 3D-FRONT)
# Modelul primește imaginea unui SINGUR obiect de mobilier (render 3D-FUTURE) și
# trebuie să-i identifice stilul + categoria și să producă o interogare de înlocuire.
# Pentru a reutiliza fără modificări metricile și taxonomy.py, schema
# folosește ACELEAȘI chei ca sarcina de scenă — câmpul `room_style` conține aici
# stilul de design al OBIECTULUI (echivalent: stilul camerei căreia i s-ar potrivi),
# iar `selected_category` categoria obiectului. Enum-urile sunt identice (inclusiv
# momeala "Art Deco" pentru proxy-ul de halucinație).
OBJECT_ANALYSIS_PROMPT = """You are shown a single rendered furniture item on a plain background.

Identify its design style and category, then describe the ideal replacement piece.
Respond in JSON format only:
{
  "room_style": "<the furniture's design style; one of: Modern, Minimalist, Nordic, Industrial, Art Deco, European Traditional, Southeast Asian, Chinese Traditional, Japanese, Luxury, Kids, Mixed>",
  "selected_object": "<brief description of the shown item, e.g. 'a beige fabric sofa'>",
  "selected_category": "<one of: Sofa, Chair, Table, Bed, Cabinet, Nightstand, Shelf, Desk, TV Stand, Floor Lamp, Ceiling Lamp, Pendant Lamp, Wardrobe, Bookcase>",
  "replacement_query": "<a 2-sentence description of the IDEAL replacement furniture piece in the same style and category, with material, shape and context>",
  "reasoning": "<why this style and category were inferred>"
}"""


def get_prompt(task_mode: str) -> str:
    """Selectează promptul după modul de sarcină ('scene' sau 'object')."""
    return OBJECT_ANALYSIS_PROMPT if task_mode == "object" else SCENE_ANALYSIS_PROMPT
