"""
Multilingual Clinical NLP Labeling Engine for Knee MRI Radiology Reports.
Accurately extracts all 12 target binary findings across English, Turkish,
Spanish, Greek, Croatian/Serbian, Cyrillic (Bulgarian/Russian), Dutch,
French, and German reports with context-aware medical negation detection.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


class MultilingualKneeLabeler:
    """
    Expert clinical rule-based parser covering 9 languages:
    English, Turkish, Spanish, Greek, Croatian/Serbian, Cyrillic, Dutch, French, German.
    Processes radiology reports sentence by sentence, evaluating pathology mentions
    against anatomical compartments and negation contexts.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # ---------------------------------------------------------------------
        # Language Identification Keywords
        # ---------------------------------------------------------------------
        self.lang_indicators = {
            "cyrillic": re.compile(r"[\u0400-\u04FF]"),
            "greek": re.compile(r"[\u0370-\u03FF]"),
            "turkish": re.compile(r"[ğüşöçıİĞÜŞÖÇ]|\b(bulgular|menisküs|çapraz|yırtık|eklem|sonuç)\b", re.IGNORECASE),
            "french": re.compile(r"\b(constatations|ménisque|épanchement|rotulienne|croisé|chondropathie|antérieur|postérieur)\b", re.IGNORECASE),
            "spanish": re.compile(r"\b(hallazgos|menisco|cruzado|derrame|condromalacia|rodilla|rotura|líquido)\b", re.IGNORECASE),
            "german": re.compile(r"\b(befund|kreuzband|meniskus|gelenkerguss|knochenmarködem|beurteilung)\b", re.IGNORECASE),
            "croatian": re.compile(r"\b(zglob|hrskavic|izljev|medijaln|lateraln|tetiv|prijelom|ruptur|edem|koljen|križn|uredn)\b", re.IGNORECASE),
            "dutch": re.compile(r"\b(bevindingen|kraakbeen|kruisband|scheur|gewricht|vocht)\b", re.IGNORECASE),
            "english": re.compile(r"\b(impression|findings|knee|meniscus|cruciate|effusion|tear|intact)\b", re.IGNORECASE),
        }

        # ---------------------------------------------------------------------
        # Negation Patterns per Language
        # ---------------------------------------------------------------------
        self.neg_prefix = re.compile(
            r"(?:"
            # English
            r"\b(?:no|without|absence\s+of|unremarkable|negative\s+for|rules?\s+out|free\s+of|denies)\b|"
            # French
            r"\b(?:pas\s+de|sans|absence\s+d['e]|aucun[e]?|intégrité\s+d[eulu']?|respect\s+d[eulu']?)\b|"
            # Spanish
            r"\b(?:sin|no|ausencia\s+de|descartad[oa]|negativo\s+para)\b|"
            # German
            r"\b(?:kein[e]?|ohne|ausschließen|fehlt)\b|"
            # Turkish
            r"\b(?:olmayan|bulunmayan|yok)\b|"
            # Cyrillic
            r"(?:няма\s+данни\s+за|няма|без\s+данни\s+за|без|отсъства|не\s+се\s+установява)|"
            # Greek
            r"(?:χωρίς|δεν\s+παρατηρείται|δεν\s+σημειώνονται|δεν\s+σηµειώνονται|χωρίς\s+ευρήματα)|"
            # Croatian
            r"\b(?:nema|bez|ne\s+vidi\s+se|nije\s+vidljivo|isključuje\s+se)\b|"
            # Dutch
            r"\b(?:geen|zonder)\b"
            r")",
            re.IGNORECASE,
        )

        self.neg_suffix = re.compile(
            r"(?:"
            # English
            r"\b(?:is\s+|are\s+)?(?:intact|normal|preserved|unremarkable|uninjured|not\s+torn|free\s+of\s+tear)\b|"
            # French
            r"\b(?:est\s+|sont\s+)?(?:intact[es]?|normal[es]?|respecté[es]?|d[' ]aspect\s+normal|sans\s+particularité|sans\s+fissure|indemne)\b|"
            # Spanish
            r"\b(?:está[n]?\s+|son\s+)?(?:íntegr[oa]s?|normal(?:es)?|intact[oa]s?|conservad[oa]s?|sin\s+rotura|sin\s+alteraciones|continu[oa]s?)\b|"
            # German
            r"\b(?:intakt|regelrecht|unauffällig|ohne\s+Riss|erhalten)\b|"
            # Turkish
            r"\b(?:izlenmedi|saptanmadı|bulunmamaktadır|yoktur|normaldir|intakttır|korunmuştur|ayırt\s+edilmedi|doğaldır|gözlenmedi)\b|"
            # Cyrillic
            r"(?:непроменен[ао]?|съхранен[ао]?|интактен|в\s+норма)|"
            # Greek
            r"(?:φυσιολογικ[όάήός]?|ακέραι[οςα]?|εντός\s+φυσιολογικών\s+ορίων)|"
            # Croatian
            r"\b(?:uredan|uredna|uredno|urednog|očuvan|očuvana|održan|održana|održanog|intaktan|intaktna)\b|"
            # Dutch
            r"\b(?:intact|normaal|onopvallend)\b"
            r")",
            re.IGNORECASE,
        )

        # ---------------------------------------------------------------------
        # Specific Finding Patterns
        # ---------------------------------------------------------------------

        # 1. ACL
        self.acl_kw = re.compile(
            r"(?:\b(?:ACL|LCA|ÖÇB|VKB|ПКВ|PKL)\b|"
            r"anterior\s+cruciate\s+ligament|"
            r"ligament\s+croisé\s+antérieur|"
            r"ligamento\s+cruzado\s+anterior|"
            r"ön\s+çapraz\s+bağ|"
            r"vorderes?\s+Kreuzband|"
            r"voorste\s+kruisband|"
            r"предна\s+кръстна\s+връзка|"
            r"πρόσθι(?:ος|ου)\s+χιαστ(?:ός|ού)|"
            r"prednji\s+križni\s+ligament|prednjeg\s+križnog\s+ligamenta)",
            re.IGNORECASE,
        )
        self.tear_kw = re.compile(
            r"(?:\b(?:tears?|ruptures?|sprains?|partial\s+tear|complete\s+tear|avulsion|detachment|disruption|torn)\b|"
            r"\b(?:roturas?|desgarros?|rupturas?|esguinces?|avulsión|desinserción)\b|"
            r"\b(?:déchirures?|entorses?|fissures?|désinsertion|ruptures?)\b|"
            r"\b(?:yırtık(?:lar)?|kopma|rüptür|devamsızlık|seçilememiştir)\b|"
            r"\b(?:Riss(?:e)?|Ruptur(?:en)?|Zerrung(?:en)?|Teilruptur|Ausriss)\b|"
            r"\b(?:scheuren?|ruptuur|inscheuring)\b|"
            r"(?:руптур[аи]|разкъсване|скъсване|лезия)|"
            r"(?:ρήξη|ρηξη|ασυνέχεια\s+των\s+ινών)|"
            r"\b(?:ruptur[ae]|prijelom|lezija|rascjep)\b)",
            re.IGNORECASE,
        )

        # 2. MCL
        self.mcl_kw = re.compile(
            r"(?:\b(?:MCL|LCM|LLI|MKL|МКЛ)\b|"
            r"medial\s+collateral\s+ligament|"
            r"ligament\s+collatéral\s+médial|"
            r"ligament\s+latéral\s+interne|"
            r"ligamento\s+colateral\s+medial|"
            r"ligamento\s+lateral\s+interno|"
            r"medial\s+kollateral\s+ligaman|iç\s+yan\s+bağ|"
            r"mediales?\s+Kollateralband|Innenband|"
            r"mediale\s+collaterale\s+band|binnenste\s+knieband|"
            r"медиален\s+колатерален\s+лигамент|"
            r"έσω\s+πλάγι(?:ος|ου)\s+σύνδεσμ(?:ος|ου)|"
            r"medijalni\s+kolateralni\s+ligament|medijalnog\s+kolateralnog\s+ligamenta)",
            re.IGNORECASE,
        )
        self.mcl_tear_kw = re.compile(
            r"(?:\b(?:tears?|sprains?|ruptures?|injur(?:y|ies)|strains?|disruption)\b|"
            r"\b(?:roturas?|esguinces?|desgarros?|rupturas?|lesión|lesiones)\b|"
            r"\b(?:déchirures?|entorses?|désinsertion|ruptures?|lésions?)\b|"
            r"\b(?:yırtık(?:lar)?|sprain|gerilme|rüptür|ödematöz\s+görünüm|ödemli)\b|"
            r"\b(?:Riss(?:e)?|Ruptur(?:en)?|Läsion(?:en)?|Zerrung(?:en)?|Distorsion)\b|"
            r"\b(?:scheuren?|ruptuur|verrekking)\b|"
            r"(?:руптур[аи]|разкъсване|скъсване|лезия|дисторзио)|"
            r"(?:ρήξη|ρηξη|ασυνέχεια)|"
            r"\b(?:ruptur[ae]|lezija|istegnuće|sprain)\b)",
            re.IGNORECASE,
        )

        # 3. Meniscus Shared
        self.med_men_kw = re.compile(
            r"(?:\b(?:MM)\b|"
            r"medial\s+meniscus|"
            r"ménisque\s+médial|ménisque\s+interne|"
            r"menisco\s+medial|menisco\s+interno|"
            r"medial\s+menisküs|medialde\s+menisküs|iç\s+menisküs|"
            r"Innenmeniskus|"
            r"mediale\s+meniscus|binnenmeniscus|"
            r"медиален\s+менискус|"
            r"έσω\s+µηνίσκ(?:ος|ου)|έσω\s+μηνίσκ(?:ος|ου)|"
            r"medijalni\s+menisk|medijalnog\s+meniska)",
            re.IGNORECASE,
        )

        self.lat_men_kw = re.compile(
            r"(?:\b(?:LM)\b|"
            r"lateral\s+meniscus|"
            r"ménisque\s+latéral|ménisque\s+externe|"
            r"menisco\s+lateral|menisco\s+externo|"
            r"lateral\s+menisküs|lateralde\s+menisküs|dış\s+menisküs|"
            r"Außenmeniskus|"
            r"laterale\s+meniscus|buitenmeniscus|"
            r"латерален\s+менискус|"
            r"έξω\s+µηνίσκ(?:ος|ου)|έξω\s+μηνίσκ(?:ος|ου)|"
            r"lateralni\s+menisk|lateralnog\s+meniska)",
            re.IGNORECASE,
        )

        self.both_menisci_kw = re.compile(
            r"(?:"
            r"both\s+menisc(?:i|uses?)|bimeniscal|ambos\s+meniscos|deux\s+ménisques|"
            r"menisco\s+(?:interno|medial)\s+y\s+(?:externo|lateral)|"
            r"medial\s+and\s+lateral\s+menisc(?:i|uses?)|"
            r"oba\s+meniska|двата\s+мениска|her\s+iki\s+menisküs|"
            r"beide\s+menisken|beide\s+menisci"
            r")",
            re.IGNORECASE,
        )

        self.meniscus_tear_kw = re.compile(
            r"(?:\b(?:tears?|torn|bucket\s+handle|root\s+tear|radial\s+tear|horizontal\s+tear|complex\s+tear|maceration|extrusion|flap\s+tear|recurrent\s+tear|low-grade\s+tear(?:ing)?)\b|"
            r"\b(?:roturas?|desgarros?|fisuras?|fissuras?|asa\s+de\s+cubo|amputación|maceración)\b|"
            r"\b(?:déchirures?|fissures?|ruptures?|anse\s+de\s+seau|désinsertion|macération)\b|"
            r"\b(?:yırtık(?:lar)?|rüptür|kova\s+sapı|evre\s+3|grade\s+3|grade\s+iii|ekstrüde)\b|"
            r"\b(?:Riss(?:e)?|Läsion(?:en)?|Korbhenkelriss|Meniskusriss)\b|"
            r"\b(?:scheuren?|ruptuur|fissuur)\b|"
            r"(?:руптур[аи]|разкъсване|скъсване|фисур[аи])|"
            r"(?:ρήξη|ρηξη)|"
            r"\b(?:ruptur[ae]|rascjep|fisure|ekstruzij[ae])\b)",
            re.IGNORECASE,
        )

        # Tricompartmental OA
        self.tricomp_oa_kw = re.compile(
            r"(?:"
            r"(?:tricompartmental|all\s+three\s+compartm|all\s+3\s+compartm|trois\s+compartiments|tres\s+compartimentos|trikompartm)\w*\s*(?:osteoarthritis|chondrosis|chondral\s+thinning|degenerative|wear|arthrosis|oa)?|"
            r"artrosis\s+tricompartimental|gonartrosis\s+tricompartimental|"
            r"arthrose\s+tricompartimentale|gonarthrose\s+tricompartimentale|"
            r"trikompartmantal\s+osteoartrit|"
            r"trikompartimentalna\s+artroza|"
            r"трикомпартментна\s+артроза"
            r")",
            re.IGNORECASE,
        )

        # 5. Medial Osteoarthritis
        self.medial_oa_kw = re.compile(
            r"(?:"
            r"(?:medial\s+(?:compartment|tibiofemoral)?\s*(?:osteoarthritis|osteoarthrosis|chondrosis|chondropathy|cartilage\s+(?:wear|loss|defect|thinning|fissur)|joint\s+space\s+narrowing|osteophytes?|pincement))|"
            r"(?:(?:osteoarthritis|chondrosis|cartilage\s+(?:defect|loss|thinning))\s+.*?\bmedial\b)|"
            r"(?:condropat[íi]a.*?(?:compartimento\s+)?femorotibial\s+medial|[úu]lceras?\s+condral(?:es)?.*?(?:medial|c[óo]ndilo\s+medial|platillo.*?medial)|condropat[íi]a.*?medial|artrosis\s+(?:femorotibial\s+)?medial|gonartrosis\s+medial|pinzamiento\s+medial|oa\s+femorotibial\s+medial)|"
            r"(?:arthrose\s+(?:fémoro-tibiale\s+)?médiale|gonarthrose\s+médiale|chondropathie\s+(?:fémoro-tibiale\s+)?médiale|pincement\s+médial)|"
            r"(?:mediale\s+Gonarthrose|medialer\s+Knorpelschaden|medialen\s+Gelenkspalt)|"
            r"(?:medial\s+femorotibial\s+osteoartrit|medial\s+eklem\s+aralığında\s+daralma|medialde\s+kıkırdak\s+kaybı)|"
            r"(?:медиална\s+гонартроза|стеснение\s+на\s+медиалната)|"
            r"(?:οστεοαρθρίτιδα.*έσω\s+διαμέρισμα|εξάλειψη\s+του\s+αρθρικού\s+χόνδρου.*έσω)|"
            r"(?:hrskavic[ae]\s+medijalnog\s+(?:kompartmenta|kondila)|artroz[ae]\s+medijalnog|denudacija\s+hrskavice\s+medijalnog)|"
            r"(?:mediaal\s+femorotibiaal\s+kraakbeenlijden|kraakbeenverlies.*mediale\s+femorale)"
            r")",
            re.IGNORECASE,
        )

        # 6. Lateral Osteoarthritis
        self.lateral_oa_kw = re.compile(
            r"(?:"
            r"(?:lateral\s+(?:compartment|tibiofemoral)?\s*(?:osteoarthritis|osteoarthrosis|chondrosis|chondropathy|cartilage\s+(?:wear|loss|defect|thinning|fissur)|joint\s+space\s+narrowing|osteophytes?|pincement))|"
            r"(?:(?:osteoarthritis|chondrosis|cartilage\s+(?:defect|loss|thinning))\s+.*?\blateral\b)|"
            r"(?:condropat[íi]a.*?(?:compartimento\s+)?femorotibial\s+lateral|[úu]lceras?\s+condral(?:es)?.*?(?:lateral|c[óo]ndilo\s+lateral|platillo.*?lateral)|condropat[íi]a.*?lateral|artrosis\s+(?:femorotibial\s+)?lateral|gonartrosis\s+lateral|pinzamiento\s+lateral|oa\s+femorotibial\s+lateral)|"
            r"(?:arthrose\s+(?:fémoro-tibiale\s+)?latérale|gonarthrose\s+latérale|chondropathie\s+(?:fémoro-tibiale\s+)?latérale|pincement\s+latéral)|"
            r"(?:laterale\s+Gonarthrose|lateraler\s+Knorpelschaden|lateralen\s+Gelenkspalt)|"
            r"(?:lateral\s+femorotibial\s+osteoartrit|lateral\s+eklem\s+aralığında\s+daralma|lateralde\s+kıkırdak\s+kaybı)|"
            r"(?:латерална\s+гонартроза|стеснение\s+на\s+латералната)|"
            r"(?:οστεοαρθρίτιδα.*έξω\s+διαμέρισμα|εξάλειψη\s+του\s+αρθρικού\s+χόνδρου.*έξω)|"
            r"(?:hrskavic[ae]\s+lateralnog\s+(?:kompartmenta|kondila|platoa)|artroz[ae]\s+lateralnog|denudacija\s+hrskavice\s+lateralnog|erozije\s+hrskavica\s+lateralnog)|"
            r"(?:lateraal\s+femorotibiaal\s+kraakbeenlijden|kraakbeenverlies.*laterale\s+tibiaplateau)"
            r")",
            re.IGNORECASE,
        )

        # 7. Patellofemoral Osteoarthritis (PF OA)
        self.pf_oa_kw = re.compile(
            r"(?:"
            r"(?:patellofemoral\s+(?:compartment\s+)?(?:osteoarthritis|chondrosis|osteoarthrosis|joint\s+space\s+narrowing|cartilage\s+(?:loss|defect|thinning|fissuring)|chondropathy))|"
            r"(?:chondromalacia\s+patell(?:ae|a)|patellar\s+chondromalacia|patellar\s+chondrosis|trochlear?\s+chondrosis|trochlear?\s+cartilage\s+(?:loss|defect|wear)|patellar\s+facet\s+cartilage)|"
            r"(?:condromalacia\s+rotuliana|condromalacia\s+patelar|artrosis\s+femoropatelar|artrosis\s+fémoro-rotuliana|condropatía\s+rotuliana|oa\s+patelofemoral)|"
            r"(?:chondromalacie\s+patellaire|chondropathie\s+rétropatellaire|arthrose\s+fémoro-patellaire|chondropathie\s+rotulienne)|"
            r"(?:Chondromalacia\s+patellae|femoropatellare\s+Arthrose|Retropatelläre\s+Chondropathie)|"
            r"(?:kondromalazi\s+patella|patellofemoral\s+osteoartrit|patellar\s+kondromalazi)|"
            r"(?:пателофеморална\s+артроза|хондромалация\s+на\s+пателата)|"
            r"(?:οστεοαρθρίτιδα\s+επιγονατιδομηριαίας|χόνδρος\s+της\s+επιγονατίδας.*οστεοαρθρίτιδας)|"
            r"(?:pf\s+artrotske|hondromalacij[ae]\s+(?:iv|iii|ii|i)?.*(?:patele|trohleje|interfasetarno)|fp\s+zgloba.*osteofit)|"
            r"(?:femoropatellair\s+kraakbeenlijden|patellofemorale\s+artrose)|"
            r"(?:chondromalacia.*?(?:patell|trochle)|(?:cartilage\s+(?:defect|loss|wear|thinning|fissur)|osteochondral\s+defect).*?(?:patell|trochle)|(?:patell|trochle).*?(?:cartilage\s+(?:defect|loss|wear|thinning|fissur)|chondromalacia))"
            r")",
            re.IGNORECASE,
        )

        # 8. Joint Effusion
        self.effusion_kw = re.compile(
            r"(?:"
            r"(?:joint\s+effusion|knee\s+effusion|suprapatellar\s+effusion|intraarticular\s+fluid|hemarthrosis|knee\s+joint\s+effusion|large\s+effusion|moderate\s+effusion)\b|"
            r"(?:derrame\s+articular|derrame\s+sinovial|derrame\s+suprapatelar|derrame\s+en\s+la\s+rodilla|hemartros|líquido\s+intraarticular)\b|"
            r"(?:épanchement\s+intra-articulaire|épanchement\s+articulaire|hydarthrose|épanchement\s+synovial|hémarthrose)\b|"
            r"(?:Gelenkerguss|Gelenkserguss|Kniegelenkserguss|Hämarthros)\b|"
            r"(?:eklem\s+effüzyonu|eklem\s+içi\s+sıvı|suprapatellar\s+bursada\s+(?:yaygın\s+)?sıvı|sıvı\s+artışı|eklem\s+içerisinde\s+sıvı)\b|"
            r"(?:ставен\s+излив|ексудат\s+в\s+ставата|течност\s+в\s+ставата)|"
            r"(?:συλλογή[ς]?\s+υγρού|ενδαρθρικ[όή]\s+υγρ[όού]|ενθαρθρικής\s+συλλογής\s+υγρού)|"
            r"(?:zglobni\s+izljev|izljev\s+u\s+zglob|opsežan\s+zglobni\s+izljev|izljev\s+u\s+suprapatelarnom)\b|"
            r"(?:gewrichtsvocht|hydrops|vochtophoping\s+in\s+het\s+knie)\b"
            r")",
            re.IGNORECASE,
        )

        # 9. Synovitis
        self.synovitis_kw = re.compile(
            r"(?:"
            r"(?:synovitis|synovial\s+hypertrophy|synovial\s+thickening|synovial\s+proliferation|pannus|reizsynovialitis)\b|"
            r"(?:sinovitis|engrosamiento\s+sinovial|hipertrofia\s+sinovial|proliferación\s+sinovial)\b|"
            r"(?:synovite|épaississement\s+synovial|hypertrophie\s+synoviale)\b|"
            r"(?:Synovitis|Synovialitis|synoviale\s+Verdickung)\b|"
            r"(?:sinovit|sinovyal\s+kalınlaşma|sinovyal\s+hipertrofi)\b|"
            r"(?:синовит|задебеляване\s+на\s+синовията)|"
            r"(?:συνοβίτιδα|συνοβιτιδα|πάχυνση\s+του\s+υμένα)|"
            r"(?:sinovitis|proliferacij[au]\s+sinovije|zadebljanje\s+sinovije)\b|"
            r"(?:verdikkingen\s+van\s+(?:het\s+)?synovium)\b"
            r")",
            re.IGNORECASE,
        )

        # 10. Baker's Cyst
        self.baker_kw = re.compile(
            r"(?:"
            r"(?:baker'?s?\s+cyst|popliteal\s+cyst|gastrocnemius-semimembranosus\s+(?:bursa|cyst))\b|"
            r"(?:quiste\s+de\s+baker|quiste\s+poplíteo|bursa\s+gastrocnemio-semimembranosa)\b|"
            r"(?:kyste\s+de\s+baker|kyste\s+poplité)\b|"
            r"(?:Baker-Zyste|Bakerzyste|popliteale\s+Zyste)\b|"
            r"(?:Baker\s+kisti|popliteal\s+kist)\b|"
            r"(?:киста\s+на\s+бейк[ъе]р|бекерова\s+киста|поплитеална\s+киста)|"
            r"(?:κύστη\s+baker|κυστη\s+baker|ιγνυακή\s+κύστη)|"
            r"(?:baker(?:ova)?\s+cista|poplitealna\s+cista)\b|"
            r"(?:popliteale\s+cyste|bakerse\s+cyste)\b"
            r")",
            re.IGNORECASE,
        )

        # 11. Bone Contusion
        self.contusion_kw = re.compile(
            r"(?:"
            r"(?:bone\s+(?:contusion|bruise|marrow\s+edema)|trabecular\s+microfracture|bone\s+edema)\b|"
            r"(?:contusión\s+ósea|edema\s+óseo|edema\s+de\s+médula\s+ósea|edema\s+trabecular|hematoma\s+óseo)\b|"
            r"(?:contusion\s+osseuse|œdème\s+osseux|œdème\s+médullaire|oedème\s+osseux|oedeme\s+osseux)\b|"
            r"(?:Knochenmarködem|Knochenkontusion|Bone\s+Bruise|Knochenödem)\b|"
            r"(?:kemik\s+iliği\s+ödemi|kemik\s+kontüzyonu|trabeküler\s+ödem|kontüzyonel\s+kemik\s+iliği)\b|"
            r"(?:костно-мозъчен\s+едем|костна\s+контузия|костномозъчен\s+оток)|"
            r"(?:οίδημα\s+του\s+οστικού\s+μυελού|οστικό[ς]?\s+μώλωπ[αας]|οστικο\s+μωλωπα)|"
            r"(?:koštani\s+edem|kostani\s+edem|edem\s+koštane\s+srži|kontuzija\s+kosti)\b|"
            r"(?:botoedeem|botcontusie|beenmergoedeem)\b"
            r")",
            re.IGNORECASE,
        )

        # 12. Fracture
        self.fracture_kw = re.compile(
            r"(?:"
            r"(?:fractures?|cortical\s+disruption|tibial\s+plateau\s+fractures?|patellar\s+fractures?|avulsion\s+fractures?|subchondral\s+(?:insufficiency\s+)?fractures?|microfractures?|osteochondral\s+(?:impaction\s+)?fractures?|segond\s+fractures?)\b|"
            r"(?:fracturas?|línea\s+de\s+fractura|hundimiento\s+cortical|fractura\s+por\s+avulsión|fracturas?\s+subcondral(?:es)?)\b|"
            r"(?:fractures?|trait\s+de\s+fracture|arrachement\s+osseux|enfoncement\s+osseux|fractures?\s+sous-chondrale[s]?)\b|"
            r"(?:Fraktur(?:en)?|Knochenfraktur(?:en)?|Fissur\s+des\s+Knochens|knöcherner\s+Ausriss|subchondrale\s+Fraktur(?:en)?)\b|"
            r"(?:kırık(?:lar)?|fraktür(?:ler)?|kortikal\s+devamsızlık|subartiküler\s+fraktür|avülsiyon\s+kırığı)\b|"
            r"(?:фрактур[аи]|счупване|кортикална\s+деструкция|остеохондрална\s+фрактура)|"
            r"(?:κάταγμα|καταγμα|κατάγματα|μικροδοκιδώδη\s+κατάγματα)|"
            r"(?:prijelom(?:i)?|fraktur[ae]|subhondralni\s+prijelom)\b|"
            r"(?:botbreuk(?:en)?|fractuur|fracturen|subchondrale\s+fractuur|impactiefractuur)\b"
            r")",
            re.IGNORECASE,
        )

    def detect_language(self, text: str) -> str:
        """Detects primary language of the radiology report."""
        if not text or not isinstance(text, str):
            return "unknown"
        norm = self._normalize_text(text)
        for lang, pat in self.lang_indicators.items():
            if pat.search(norm):
                return lang
        return "english"

    def _normalize_text(self, text: str) -> str:
        """Normalizes Unicode micro symbols, carriage returns, etc."""
        return text.replace("\u00b5", "μ").replace("\r", "\n")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits report into individual clauses/sentences for context isolation."""
        clean = self._normalize_text(text)
        raw_parts = re.split(r"[\n\.;•>–\t]+", clean)
        return [p.strip() for p in raw_parts if len(p.strip()) > 2]

    def _is_negated(self, sentence: str, match_start: int, match_end: int) -> bool:
        """
        Checks whether a matched pathology is preceded or followed by a negation
        within the same clause.
        """
        prefix = sentence[:match_start].strip()
        suffix = sentence[match_end:].strip()

        # Check prefix negation (up to 45 characters before match)
        prefix_window = prefix[-45:] if len(prefix) > 45 else prefix
        if self.neg_prefix.search(prefix_window):
            return True

        # Check suffix negation (up to 45 characters after match)
        suffix_window = suffix[:45] if len(suffix) > 45 else suffix
        if self.neg_suffix.search(suffix_window):
            return True

        # Check intrameniscal signal grade 1/2 without articular surface disruption
        lower_sent = sentence.lower()
        if any(g in lower_sent for g in ["grade 1", "grade 2", "grade i", "grade ii", "grado 1", "grado 2", "evre 1", "evre 2"]):
            if any(term in lower_sent for term in ["sin rotura", "no tear", "sans fissure", "yırtık izlenmedi", "intact", "without tear", "без разкъсване"]):
                return True

        return False

    def parse_report_with_details(self, report_text: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Parses a single report and returns:
        1. binary predictions {finding: 0 or 1}
        2. evidence spans {finding: evidence_sentence}
        """
        if not report_text or not isinstance(report_text, str) or report_text.strip() == "":
            return {label: 0 for label in TARGET_LABELS}, {label: "" for label in TARGET_LABELS}

        report_clean = self._normalize_text(report_text)
        sentences = self._split_into_sentences(report_clean)
        results = {label: 0 for label in TARGET_LABELS}
        evidence = {label: "" for label in TARGET_LABELS}

        # Check tricompartmental OA across the entire report
        for sent in sentences:
            if self.tricomp_oa_kw.search(sent):
                if not self.neg_prefix.search(sent.lower()):
                    results["Medial OA"] = 1
                    results["Lateral OA"] = 1
                    results["PF OA"] = 1
                    evidence["Medial OA"] = sent
                    evidence["Lateral OA"] = sent
                    evidence["PF OA"] = sent

        for sent in sentences:
            sent_clean = sent.lower()

            # --- ACL Check ---
            acl_m = self.acl_kw.search(sent)
            if acl_m:
                tear_m = self.tear_kw.search(sent)
                if tear_m:
                    if not self._is_negated(sent, tear_m.start(), tear_m.end()):
                        results["ACL"] = 1
                        evidence["ACL"] = sent

            # --- MCL Check ---
            mcl_m = self.mcl_kw.search(sent)
            if mcl_m:
                tear_m = self.mcl_tear_kw.search(sent)
                if tear_m:
                    if not self._is_negated(sent, tear_m.start(), tear_m.end()):
                        results["MCL"] = 1
                        evidence["MCL"] = sent

            # --- Both Menisci Rule ---
            both_m = self.both_menisci_kw.search(sent)
            if both_m:
                tear_m = self.meniscus_tear_kw.search(sent)
                if tear_m:
                    if not self._is_negated(sent, tear_m.start(), tear_m.end()):
                        results["Medial Meniscus"] = 1
                        results["Lateral Meniscus"] = 1
                        evidence["Medial Meniscus"] = sent
                        evidence["Lateral Meniscus"] = sent

            # --- Medial Meniscus Check ---
            med_m = self.med_men_kw.search(sent)
            if med_m:
                tear_m = self.meniscus_tear_kw.search(sent)
                if tear_m:
                    if not self._is_negated(sent, tear_m.start(), tear_m.end()):
                        results["Medial Meniscus"] = 1
                        evidence["Medial Meniscus"] = sent

            # --- Lateral Meniscus Check ---
            lat_m = self.lat_men_kw.search(sent)
            if lat_m:
                tear_m = self.meniscus_tear_kw.search(sent)
                if tear_m:
                    if not self._is_negated(sent, tear_m.start(), tear_m.end()):
                        results["Lateral Meniscus"] = 1
                        evidence["Lateral Meniscus"] = sent

            # --- Medial OA ---
            moa_m = self.medial_oa_kw.search(sent)
            if moa_m:
                if not self._is_negated(sent, moa_m.start(), moa_m.end()):
                    results["Medial OA"] = 1
                    evidence["Medial OA"] = sent

            # --- Lateral OA ---
            loa_m = self.lateral_oa_kw.search(sent)
            if loa_m:
                if not self._is_negated(sent, loa_m.start(), loa_m.end()):
                    results["Lateral OA"] = 1
                    evidence["Lateral OA"] = sent

            # --- PF OA ---
            pfoa_m = self.pf_oa_kw.search(sent)
            if pfoa_m:
                if not self._is_negated(sent, pfoa_m.start(), pfoa_m.end()):
                    results["PF OA"] = 1
                    evidence["PF OA"] = sent

            # --- Effusion ---
            eff_m = self.effusion_kw.search(sent)
            if eff_m:
                is_trace = any(t in sent_clean for t in [
                    "trace", "physiologic", "minime", "fisiológico",
                    "fizyolojik", "discreto", "minimal", "minimo", "geringer",
                    "nema značajnijeg", "bez značajnije", "manja količina"
                ])
                if not is_trace and not self._is_negated(sent, eff_m.start(), eff_m.end()):
                    results["Effusion"] = 1
                    evidence["Effusion"] = sent

            # --- Synovitis ---
            syn_m = self.synovitis_kw.search(sent)
            if syn_m:
                if not self._is_negated(sent, syn_m.start(), syn_m.end()):
                    results["Synovitis"] = 1
                    evidence["Synovitis"] = sent

            # --- Baker's Cyst ---
            bak_m = self.baker_kw.search(sent)
            if bak_m:
                is_trace = any(t in sent_clean for t in ["trace", "minimo", "discreto", "minimal"])
                if not is_trace and not self._is_negated(sent, bak_m.start(), bak_m.end()):
                    results["Baker's"] = 1
                    evidence["Baker's"] = sent

            # --- Bone Contusion ---
            con_m = self.contusion_kw.search(sent)
            if con_m:
                if not self._is_negated(sent, con_m.start(), con_m.end()):
                    results["Contusion"] = 1
                    evidence["Contusion"] = sent

            # --- Fracture ---
            fra_m = self.fracture_kw.search(sent)
            if fra_m:
                if not self._is_negated(sent, fra_m.start(), fra_m.end()):
                    results["Fracture"] = 1
                    evidence["Fracture"] = sent

        return results, evidence

    def parse_report(self, report_text: str) -> Dict[str, int]:
        """
        Parses a single report and returns binary predictions {finding: 0 or 1}
        for all 12 conditions.
        """
        results, _ = self.parse_report_with_details(report_text)
        return results
