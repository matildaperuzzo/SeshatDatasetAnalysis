import numpy as np

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

miltech_mapping_api = {
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
        "elephants": 2,
        "camels": 2,
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

social_complexity_mapping_api = {

    "Scale":{ "polity-populations":1,
             "polity-territories":1,
             "population-of-the-largest-settlements":1
             },
             
    "Hierarchy": {"administrative-levels": 1,
                # "religious-levels": 1,
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

ideology_mapping_api = {
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

    "HumanSacrifice" : {"human-sacrifice" : 1
                }

}

luxury_mapping_api = {"LuxuryItems" : {"lux-precious-metal" : 1,
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


religion_mapping_api = {"Religion" : {"polity-religion-genuses" : 1,
                                  "polity-religion-families" : 1,
                                  "polity-religions" : 1
                                 }
        }



miltech_mapping_polaris = {
    'Long_walls': {
        'long_wall': 1
    },
    'Metal': {
        'copper': 1,
        'bronze': 2,
        'iron': 3,
        'steel': 4
    },
    'Project': {
        'javelin': 1,
        'atlatl': 2,
        'sling': 3,
        'self_bow': 4,
        'composite_bow': 5,
        'crossbow': 6,
        'tension_siege_engine': 7,
        'sling_siege_engine': 8,
        'gunpowder_siege_artillery': 9,
        'handheld_firearm': 10
    },
    'Weapon': {
        'war_club': 1,
        'battle_axe': 1,
        'dagger': 1,
        'sword': 1,
        'spear': 1,
        'polearm': 1
    },
    'Armor_max': {
        'wood_bark_etc': 1,
        'leather_cloth': 2,
        'chainmail': 3,
        'scaled_armor': 3,
        'laminar_armor': 4,
        'plate_armor': 4
    },
    'Armor_mean': {
        'shield': 1,
        'helmet': 1,
        'breastplate': 1,
        'limb_protection': 1
    },
    'Other Animals': {
        'elephant': 2,
        'camel': 2
    },
    'Animals': {
        'donkey': 1,
        'elephant': 2,
        'camel': 2,
        'horse': 3
    },
    'Fortifications': {
        'fortified_camp': 1,
        'complex_fortification': 1,
        'modern_fortification': 1
    },
    'Fortifications_max': {
        'settlements_in_a_defensive_position': 1,
        'earth_rampart': 2,
        'stone_walls_non_mortared': 3,
        'wooden_palisade': 4,
        'stone_walls_mortared': 5
    },
    'Surroundings': {
        'ditch': 5,
        'moat': 5
    }
}

social_complexity_mapping_polaris = {
    'Scale': {'polity_population': 1,
              'polity_territory': 1,
              'population_of_the_largest_settlement': 1},
    'Hierarchy': {'administrative_level': 1,
                  'military_level': 1,
                  'settlement_hierarchy': 1},
    'Government': {'professional_military_officer': 1,
                   'professional_soldier': 1,
                   'professional_priesthood': 1,
                   'full_time_bureaucrat': 1,
                   'professional_lawyer': 1,
                   'examination_system': 1,
                   'court': 1,
                   'judge': 1,
                   'formal_legal_code': 1,
                   'merit_promotion': 1,
                   'specialized_government_building': 1},
    'Infrastructure': {'irrigation_system': 1,
                       'drinking_water_supply_system': 1,
                       'market': 1,
                       'food_storage_site': 1,
                       'road': 1,
                       'bridge': 1,
                       'canal': 1,
                       'port': 1,
                       'mines_or_quarry': 1,
                       'courier': 1,
                       'postal_station': 1,
                       'general_postal_service': 1},
    'Information': {'mnemonic_device': 1,
                    'script': 1,
                    'lists_tables_and_classification': 1,
                    'phonetic_alphabetic_writing': 1,
                    'non_phonetic_writing': 1,
                    'written_record': 1,
                    'nonwritten_record': 1,
                    'calendar': 1,
                    'scientific_literature': 1,
                    'sacred_text': 1,
                    'history': 1,
                    'religious_literature': 1,
                    'fiction': 1,
                    'practical_literature': 1,
                    'philosophy': 1},
    'Money': {'article': 1,
              'token': 2,
              'precious_metal': 3,
              'foreign_coin': 4,
              'indigenous_coin': 5,
              'paper_currency': 6}}



ideology_mapping_polaris = {
    'MSP': {'moralizing_supernatural_concern_is_primary': 1,
            'moralizing_enforcement_is_certain': 1,
            'moralizing_enforcement_is_broad': 1,
            'moralizing_enforcement_is_targeted': 1,
            'moralizing_enforcement_of_rulers': 1,
            'moralizing_religion_adopted_by_elites': 1,
            'moralizing_religion_adopted_by_commoners': 1,
            'moralizing_enforcement_in_afterlife': 1,
            'moralizing_enforcement_in_this_life': 1,
            'moralizing_enforcement_is_agentic': 1},
    "HumanSacrifice": {"human_sacrifice": 1
                       }
}


luxury_mapping_polaris = {
    'LuxuryItems': {
        'lux_precious_metal': 1,
        'luxury_fabrics': 1,
        'luxury_manufactured_goods': 1,
        'luxury_spices_incense_and_dyes': 1,
        'luxury_drink_alcohol': 1,
        'luxury_glass_goods': 1,
        'lux_fine_ceramic_wares': 1,
        'lux_precious_stone': 1,
        'lux_statuary': 1,
        'luxury_food': 1,
        'other_luxury_personal_items': 1}
}


religion_mapping_polaris = {
    'Religion': {'polity_religion_genus': 1,
                 'polity_religion_family': 1,
                 'polity_religion': 1}
}






def get_mapping(category : str, scheme : str = 'polaris'):
    """
    Get the variable mapping to aggregate data belonging to a specific category.

    Parameters:
    category (str): variable category. Must be one of 'miltech', 'social complexity',
        'ideology', 'luxury', or 'religion'. The shorthand 'sc' can be used for 'social_complexity'.
    scheme (str): Variable naming scheme to use. Must be one of:
        'polaris' -- use the variable naming scheme used with the Polaris release (default)
        'api' -- use the variable naming scheme of the web API (deprecated)
    TODO: add support for Equinox variable names

    Returns:
    A dict of dicts with mappings between individual and aggregated variables.
    """
    
    if scheme != 'polaris' and scheme != 'api':
        raise BaseException('Unknown variable naming scheme!')
    
    if category == 'miltech':
        if scheme == 'polaris':
            return miltech_mapping_polaris
        if scheme == 'api':
            return miltech_mapping_api
    
    elif category == 'social_complexity' or category == 'sc':
        if scheme == 'polaris':
            return social_complexity_mapping_polaris
        if scheme == 'api':
            return social_complexity_mapping_api

    if category == 'ideology':
        if scheme == 'polaris':
            return ideology_mapping_polaris
        if scheme == 'api':
            return ideology_mapping_api
    
    if category == 'luxury':
        if scheme == 'polaris':
            return luxury_mapping_polaris
        if scheme == 'api':
            return luxury_mapping_api
    
    if category == 'religion':
        if scheme == 'polaris':
            return religion_mapping_polaris
        if scheme == 'api':
            return religion_mapping_api
    
    raise BaseException(f'Unknown variable category requested ({category})')


