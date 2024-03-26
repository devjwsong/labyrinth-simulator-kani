RULE_SUMMARY = [
    [
        "Player characters: Each player has own character.",
        "Basically, each character has its name, kin, persona, and goal.",
        "Also, each player character has own traits, flaws, and items.",
        "You should consider these property of players when you interact with them and try to be consistent with them while being creative."
    ],
    [
        "Difficulty: The player party will come across many challenges and when a player character tries something that has a chance of failure, there should be a test by rolling a dice.",
        "The Goblin King should decides how difficult the test is by choosing a number between 2 and 6.",
        "Then, the Goblin King should let the player roll a die and see if the result is equal to or higher than the difficulty number.",
        "If it is, the player has succeeded to pass, otherwise the player has failed.",
        "The result 1 is always considered as failure.",
        "If the player has a trait which can improve the test, make the player roll the dice twice and take the larger value.",
        "If the player has a flaw which can hinder the test, make the player roll the dice twice and take the smaller value.",
        "If other players declare to help the test of a player, they should specify how they can help with the traits they have.",
        "Each player that is helping reduces the difficulty by 1 to a minimum of 2."
    ],
    [
        "Equipment: The players might find items or equipments throughout the Labyrinth.",
        "Players can ask the Goblin King if they can use the items, and you may decide that an item improves your test or allows you to succeed without rolling altogether.",
        "The maximum number of items a player can carry is 6.",
        "If a player wants to pick up an additional item but the inventory is full, an item should be dropped or given to another player."
    ],
    [
        "Action scenes: An action scene is a situation where the players should do something under a strict time limit.",
        "If they do nothing, the situation would change drastically.",
        "An action scene is declared by the Goblin King.",
        "After the action scene is activated, each player should describe one action to do per player.",
        "Then the Goblin King asks the player to test if the action is successful via dice roll.",
        "Unlike normal tests, the teamwork by multiple players reduces the difficulty of the task by one without any relevant traits.",
        "Each round during the action scene is about 5 seconds long, so the players should be hurry to take some actions.",
        "After each player has made the tests, the Goblin King should decide how the scene has gone as an outcome.",
        "If there are reactions from foes or other objects, the Goblin King can ask the players for another actions and tests to avoid the reaction.",
        "You should keep describing the changes after the players' action and requiring them to take another actions as a counter.",
        "When you think that the action scene has finally ended, this should be announced to the players to terminate the action scene.",
        "If you are confused with whether you should declare an action scene or a simple test, consider if there are multiple possible outcomes.",
        "If there are, declaring an action scene might be better, otherwise just a test."
    ],
    [
        "Using tables: The random tables are used during the scene if there are some objects that should be brought in or some entries should be selected for proceeding the scene.",
        "The Goblin King should first determine if the current game flow requires to use the random table, decide which table should be referred to, pick the entries depending on the requirement, and use them during the game.",
        "The sampled entries can be used for just adding the context during the gameplay, or directly updating the required game states."
    ],
    [
        "Hours: Other than the time limit during an action scene, the Labyrinth has the total time limit per each scene.",
        "The Goblin King should count the time and tell the players everytime they lose an hour.",
        "If 13 hours has passed, the players lose the game regardless of the current game state."
    ],
    [
        "NPCs: Sometimes, the players can make friends with NPCs by convincing them during the game.",
        "However, the NPCs are usually afraid of the Goblin King, they would not stay long.",
        "If a player tries to chat with an NPC, you should generate the NPC by finding the corresponding NPC's name in 'npcs' and checking its persona and goal.",
        "If there is no name which is matched with the one the player wants to talk to, you can make the kin, persona, goal, traits, and flaws for the requested NPC on the spot to generate the NPC.",
        "However, if you think the NPC requested doesn't fit the given scene or context so far, you should reject it and tell the user the reason.",
        "Referring to the generated NPC's information, you should act out the NPC's part.",
        "You might have to mimic its tone, charactertics, or behavior genuinely.",
        "Tag the NPC's line with additional \".",
        "Depending upon an NPC's flaw, they may leave the group at certain times depending on it's flaw.",
        "To see if an NPC is leaving, make the players roll a test with difficulty 4.",
        "If the test is successful, the NPC stays in the party.",
        "If the Goblin King himself appears in the scene, all NPCs will flee without any tests."
    ],
    [
        "Additional instructions: During the interactions with the players, you should keep your role as a game manager.",
        "You should keep track of the game flow and try to lead the game which should not be too far from the game flow rules.",
        "And you should keep thinking whether the current circumstance is matched with the success/failure condition so that you can wrap-up the scene."
    ]
]

# Options setting.
RESPONSE_CONSISTENCY_OPTIONS = [
    "Check the initial scene state.",
    "Check the initial player states",
    "Check the past chat history.",
    "Check the current queries.",
    "Give score."
]
RESPONSE_RELIABILITY_OPTIONS = [
    "Check the game rules.",
    "Check the initial scene state.",
    "Check the initial player states",
    "Check the past chat history.",
    "Check the current queries.",
    "Give score."
]
RESPONSE_INTERESTINGNESS_OPTIONS = [
    "Give score."
]

# Rubric setting.
CONSISTENCY_RUBRIC = {
    'question': "How consistent is the generated response to the current game progress?",
    'specifications': {
        "The generated response is consistent with the previous interactions between the game manager and players.": [
            "The model remembers the past interactions.",
            "The response is relevant to the player party's queries or requests."
        ],
        "The generated response is consistent with the current game state so far.": [
            "The model acknowledges the existing components in the current scene, such as NPCs, objects, and random table entries.",
            "The model acknowledges the existing properties of the players, such as traits, flaws, and inventories."
        ]
    },
    'notes': [
        "If the model output hallucinates any non-existing components, ignore it for this question. This will be penalized in the reliability check question."
    ],
    'max_score': 1.0,
    'min_score': 0.0
}

RELIABILITY_RUBRIC = {
    'question': "How well does the model control and manage the game reliably?",
    'specifications': {
        "The game manager fully understands the game and performs its task as a manager correctly.": [
            "The model acts while following the general game rules in Labyrinth.",
            "The model understands the scene-specific rules, instructions, and specifications  of the current scene and guides the players to proceed with the game as intended."
        ],
        "When a player tries to do something which violates the rule or fake up anything which does not exist, the game manager rejects or ignores it.": [
            "The model rejects it when the player attempts to do something which cannot be performed by a player character according to the rule.",
            "The model rejects it when the player tries to use a trait, flaw, or item which does not exist in the player.",
            "The model rejects it when the player tries to leverage or get access to non-existing environmental objects or NPCs, etc."
        ],
        "Any unexpected behavior which might hurt the players' gameplay experience or make the game flow far from intended should not be performed.": []
    },
    'notes': [
        "Note that this metric does not evaluate the quality or consistency of the response.",
        "Even if the response looks perfect, it can contain hallucinated content or the model might just let the player do an unallowed trial."
    ],
    'max_score': 1.0,
    'min_score': 0.0
}

INTERESTINGNESS_RUBRIC = {
    'question': "How interesting is the generated response?",
    'specifications': {
        "The response describes the scene funny, entertaining and specific.": [],
        "The response makes the user engaged and immersed in the game.": []
    },
    'notes': [],
    'max_score': 1.0,
    'min_score': 0.0
}

FUNCTION_RUBRICS = {
    'activate_test': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when the dice roll test is required considering the progress and game state so far.": []
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is correct.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented in a different way, it is considered wrong since it cannot be parsed from the player list."
                ],
                "'initial_difficulty' is correct.": [
                    "If the difficulty value for this test is mentioned in the scene, 'initial_difficulty' should be identical to that value.",
                    "If the difficulty value for this test is not mentioned in the scene, validate if the difficulty makes sense considering the test the player tries to do.",
                    "The difficulty value should be an integer between 2 and 6."
                ],
                "'final_difficulty' is correct.": [
                    "If the teammates help the player to do the test, 'initial_difficulty' should be reduced to 'final_difficulty'.",
                    "The traits of the teammates which help the test should have some reasonable advantages to make the test easy.",
                    "Each help from a teammate decreases the difficulty by 1 each, but 'final_difficulty' should not be less than 2."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_detection_of_improvement': {
            'question': "Does the function correctly detect the improvement/hindrance of the test?",
            'specifications': {
                "The intermediate result of the improvement detection is matched with whether the test was improved, hindered, or not affected based on the player's traits or flaws.": []
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'activate_action_scene': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when the action scene should start considering the progress and game state so far.": [
                    "If the function is called when 'is_action_scene' is already true, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'terminate_action_scene': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when the action scene should end considering the progress and game state so far.": [
                    "If the function is called when 'is_action_scene' is already false, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'create_npc': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a new NPC should be set considering the progress and game state so far.": [
                    "If 'npc_name' already exists in 'npcs' in the scene, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'npc_name' is matched with the NPC which the players encountered or tried to interact with.": [
                    "Even if the NPC already exists but the function has been called falsely, the argument should be considered correct if it is the right name in 'npcs' in the scene.",
                    "If the NPC does not exist, then the name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                ],
                "'npc_desc' is properly parsed.": [
                    "If the NPC description is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the NPC description is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the NPC name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'npc_quality': {
            'question': "Are the generated specifications satisfying?",
            'specifications': {
                "All values are following the conditions, reasonable, natural, and not contradictory with each other in the specifications.": [
                    "The trait and flaw should follow the format: {NAME} - {DESCRIPTION}.",
                    "The flaw should be one of the available options listed in the game manual."
                ],
                "All values should be consistent with other scene state components. (e.g. other NPCs, random tables, environment, etc.) without any contradiction.": [],
                "The given description does exist in the generated specification. (e.g. in persona, goal, or trait)": []
            }
        }
    },
    'add_trait': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a new trait should be added to the player considering the progress and game state so far.": [
                    "If 'trait_name' already exists in 'traits' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional trait.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented in a different way, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'trait_name' is properly parsed.": [
                    "Even if the trait already exists but the function has been called falsely, the argument should be considered correct if it is the right name in 'traits' in the player.",
                    "If the trait does not exist, then the name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                ],
                "'trait_desc' is properly parsed.": [
                    "If the trait description is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the trait description is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the trait name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'add_flaw': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a new flaw should be added to the player considering the progress and game state so far.": [
                    "If 'flaw_name' already exists in 'flaws' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional flaw.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented in a different way, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'flaw_name' is properly parsed.": [
                    "Even if the flaw already exists but the function has been called falsely, the argument should be considered correct if it is the right name in 'flaws' in the player.",
                    "If the flaw does not exist, then the name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                ],
                "'flaw_desc' is properly parsed.": [
                    "If the flaw description is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the flaw description is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the flaw name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'add_item': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a new item should be added to the player considering the progress and game state so far.": [
                    "If 'item_name' already exists in 'inventory' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional item.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented in a different way, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'item_name' is properly parsed.": [
                    "Even if the item already exists but the function has been called falsely, the argument should be considered correct if it is the right name in 'inventory' in the player.",
                    "If the item does not exist, then the name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                ],
                "'item_desc' is properly parsed.": [
                    "If the item description is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the item description is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the item name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 1.0,
            'min_score': 0.0
        }
    },
    'remove_trait': {},
    'remove_flaw': {},
    'remove_item': {},
    'use_item': {},
    'add_object': {},
    'use_environment': {},
    'use_random_table': {},
}