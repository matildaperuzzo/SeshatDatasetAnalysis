import numpy as np
import random

value_mapping = {
    "present": 1,
    "Present": 1,
    "P": 1,
    "inferred present": 0.9,
    "Inferred Present": 0.9,
    "P*": 0.9,
    "IP": 0.9,
    "inferred p": 0.9,
    "absent": 0,
    "Absent": 0,
    "A": 0,
    "inferred absent": 0.1,
    "Inferred Absent": 0.1,
    "A*": 0.1,
    "IA": 0.1,
    "inferred a": 0.1,
    "A~P": 0.5,
    "P~A": 0.5,
    "absent_to_present": 0.5,
    "present_to_absent": 0.5,
    "unknown": np.nan,
    "Unknown": np.nan,
    "U": np.nan,
    "suspected unknown": np.nan,
    "Suspected Unknown": np.nan,
    'SU' : np.nan,
    "U*": np.nan
}

PT_value_mapping = {
    'A': 0,
    'P': 1,
    'U': np.nan,
    'IA': 0.1,
    'IP': 0.9,
    'SU': np.nan,
    'None': np.nan,
    'DIS' : 0.5
}

miltech_mapping = {
        "Long_walls" : {
            "long-walls" : 1
            },
        "Metal" : {
        "coppers" : 1,
        "bronzes" : 2,
        "irons"   : 3,
        "steels"  : 4},

        "Project" : {
        "javelins"                  : 1,
        "atlatls"                    : 2,
        "slings"                    : 3,
        "self-bows"                  : 4,
        "composite-bows"             : 5,
        "crossbows"                  : 6,
        "tension-siege-engines"     : 7,
        "sling-siege-engines"       : 8,
        "gunpowder-siege-artilleries" : 9,
        "handheld-firearms"        : 10},

        "Weapon" : {
        "war-clubs" : 1,
        "battle-axes" : 1,
        "daggers" : 1,
        "swords" : 1,
        "spears" : 1,
        "polearms" : 1},

        "Armor_max" : {
        "wood-bark-etc" : 1,
        "leathers" : 2,
        "chainmails"     : 3,
        "scaled-armors"  : 3,
        "laminar-armors" : 4,
        "plate-armors"   : 4},

        "Armor_mean" : {
        "shields" : 1,
        "helmets" : 1,
        "breastplates" : 1,
        "limb-protections" : 1},

        "Other Animals" : {
        "elephants": 2,
        "camels": 2
        },

        "Animals" : {
        "donkeys": 1,
        "other-animals": 2,
        "horses": 3},

        "Fortifications" : {
        "fortified-camps" : 1,
        "complex-fortifications" : 1,
        "modern-fortifications" : 1},

        "Fortifications_max" : {
        "settlement-in-defensive-positions" : 1,
        "earth-ramparts" : 2,
        "stone-walls-non-mortared" : 3,
        "wooden-palisades" : 4,
        "stone-walls-mortared" : 5 },

        "Surroundings" : {
        "ditches" : 5,
        "moats" : 5 }
        
}

social_complexity_mapping = {

    "Scale":{ "polity-populations":1,
             "polity-territories":1,
             "population-of-the-largest-settlements":1
             },
             
    "Hierarchy": {"administrative-levels": 1,
                "religious-levels": 1,
                "military-levels": 1,
                "settlement-hierarchies": 1
                },
    "Government": {"professional-military-officers" : 1,
                   "professional-soldiers" : 1,
                   "professional-priesthoods" : 1,
                   "full-time-bureaucrats" : 1,
                   "professional-lawyers" : 1,
                   "examination-systems" : 1,
                   "courts" : 1,
                   "judges" : 1,
                   "formal-legal-codes" : 1,
                   "merit-promotions" : 1,
                   "specialized-government-buildings" : 1
                   },
    "Infrastructure": {"irrigation-systems" : 1,
                       "drinking-water-supplies" : 1,
                       "markets" : 1,
                       "food-storage-sites" : 1,
                       "roads" : 1,
                       "bridges" : 1,
                       "canals" : 1,
                       "ports" : 1,
                       "mines-or-quarries" : 1,
                       "couriers" : 1,
                       "postal-stations" : 1,
                       "general-postal-services" : 1
                       },
    "Information": {"mnemonic-devices" : 1,
                    "scripts" : 1,
                    "lists-tables-and-classifications" : 1,
                    "phonetic-alphabetic-writings" : 1,
                    "non-phonetic-writings" : 1,
                    "written-records" : 1,
                    "nonwritten-records" : 1,
                    "calendars" : 1,
                    "scientific-literatures" : 1,
                    "sacred-texts" : 1,
                    "histories" : 1,
                    "religious-literatures" : 1,
                    "fictions" : 1,
                    "practical-literatures" : 1,
                    "philosophies" : 1},
    "Money": {"articles": 1,
              "tokens": 2,
              "precious-metals" : 3,
              "foreign-coins" : 4,
              "indigenous-coins" : 5,
              "paper-currencies" : 6
              }
}

ideology_mapping = {
    'MSP' : {"moralizing-supernatural-concern-is-primary" : 1, # "PrimaryMSP"
                "moralizing-enforcement-is-certain" : 1, # "CertainMSP"
                "moralizing-enforcement-is-broad" : 1, # "BroadMSP"
                "moralizing-enforcement-is-targeted" : 1, # "TargetedMSP"
                "moralizing-enforcement-of-rulers" : 1, # "RulerMSP"
                "moralizing-religion-adopted-by-elites" : 1, # "ElitesMSP"
                "moralizing-religion-adopted-by-commoners" : 1, # "CommonersMSP"
                "moralizing-enforcement-in-afterlife" : 1, # "AfterlifeMSP"
                "moralizing-enforcement-in-this-life" : 1, # "ThislifeMSP"
                "moralizing-enforcement-is-agentic" : 1
    },
    # 'EgalID' : {"Ideological-reinforcement-of-equality" : 1,
    #             "Ideological-thought-equates-elites-and-commoners" : 1,
    #             "Ideological-thought-equates-rulers-and-commoners" : 1
    #             },
    # 'Prosoc' : {"Ideology-reinforces-prosociality" : 1,
    #             'production-of-public-goods' : 1
    #             },
    # 'Constrain' : {"Constraint-on-executive-by-government" : 1,
    #                "Constraint-on-executive-by-non-government" : 1,
    #                "Impeachment" : 1
    #             },
    "HumanSacrifice" : {"human-sacrifice" : 1
                }

}

luxury_mapping = {"LuxuryItems" : {"lux-precious-metal" : 1,
                                    "luxury-fabrics" : 1,
                                    "luxury-manufactured-goods" : 1,
                                    "luxury-spices-incense-and-dyes" : 1,
                                    "luxury-drink-alcohol" : 1,
                                    "luxury-glass-goods" : 1,
                                    "lux-fine-ceramic-wares" : 1,
                                    "lux-precious-stone" : 1,
                                    "lux-statuary" : 1,
                                    "luxury-food" : 1,
                                    "other-luxury-personal-items" : 1
                                   }
                }