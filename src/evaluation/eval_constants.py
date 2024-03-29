TASK_INSTRUCTION = [
    "<p><strong><h2>Instructions</h2></strong><br>",
    "In this task, you will see the part of the gameplay data of a text adventure game called \"Jim Henson's Labyrinth: The Adventure Game\". ",
    "For each target response, you should answer the question considering the starting game states, the past chat history, and the current queries. ",
    "This survey has been designed assuming that you have fully understood the evaluation guideline posted along with this survey. ",
    "If you have any questions or issues, feel free to contact the survey designer, whose email address has been attached to the last page of the guideline. "
]

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
FUNCTION_OPTIONS = [
    "Check the game rules.",
    "Check the current scene state.",
    "Check the current player states.",
    "Check the past chat history.",
    "Check the current queries.",
    "Give score."
]

# Rubric setting.
CONSISTENCY_RUBRIC = {
    'question': "How consistent is the target response to the current game progress, including the dialogue and the game states?",
    'specifications': {
        "The target response is consistent with the interaction between the players and the manager so far. (Past chat history + Current queries)": [
            "The model remembers the past interactions.",
            "The response is relevant to the player party's queries."
        ],
        "The target response is consistent with the updates in the scene and players so far.": [
            "The model acknowledges the existing components in the current scene, such as NPCs, objects, and random table entries.",
            "The model acknowledges the existing properties of the players, such as traits, flaws, and inventories."
        ]
    },
    'notes': [
        "The starting scene state and player states are only initialized ones when the game started. While proceeding with the game, these states might have been updated, which you should also consider for your evaluations.",
        "If the model output assumes or fakes up any non-existing components, ignore it for this question. This will be penalized in the reliability check question."
    ],
    'examples': [
        "1=The model does not follow the state at all",
        "3=The model makes a narration that is plausible but misses some components in the scene or players",
        "5=The model's response is correctly acknowledging the components in the states"
    ],
    'max_score': 5,
    'min_score': 1
}

RELIABILITY_RUBRIC = {
    'question': "How well does the model control and manage the game reliably?",
    'specifications': {
        "The game manager fully understands the game and performs its task as a manager correctly.": [
            "The model keeps the general game rules in Labyrinth.",
            "The model understands the scene-specific rules, instructions, and specifications of the current scene and guides the players to proceed with the game as intended."
        ],
        "When a player tries to do something invalid, the game manager rejects it robustly.": [
            "The model rejects it when the player attempts to do something which cannot be performed by a player character or which is not the player's task.",
            "The model rejects it when the player tries to use a trait, flaw, or item which does not exist in the player.",
            "The model rejects it when the player tries to leverage or get access to non-existing objects, NPCs, or random tables."
        ],
        "Any unexpected behavior which might hurt the players' gameplay experience or make the game flow far from intended should be penalized.": []
    },
    'notes': [
        "Note that this metric does not evaluate the quality of the response. Even if the response looks perfect, it can contain an invalid content or the model might just let the player do an unallowed trial."
    ],
    'examples': [
        "1=The model blatantly ignores the rules or is completely generous with the players' invalid moves, which makes the game go into a bad state",
        "3=The model gets some rules incorrect or accepts the players' some violations, but the game generally progresses as it should",
        "5=The model keeps the rules correctly and corrects the players' invalid or unacceptable behaviors"
    ],
    'max_score': 5,
    'min_score': 1
}

INTERESTINGNESS_RUBRIC = {
    'question': "How interesting is the generated response?",
    'specifications': {
        "The response describes the scene funny, entertaining and specific.": [],
        "The response makes the user engaged and immersed in the game.": []
    },
    'notes': [],
    'examples': [
        "1=The response is too bland, simple, or half-hearted",
        "3=The response is not highly entertaining, but at least it is not boring",
        "5=The response is so engaging and immersive that I wouldn't want to stop the game if I were a player"
    ],
    'max_score': 5,
    'min_score': 1
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
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is correct.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list."
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
            'max_score': 10,
            'min_score': 0
        },
        'correct_detection_of_improvement': {
            'question': "Does the function correctly detect the improvement/hindrance of the test?",
            'specifications': {
                "The intermediate result of the improvement detection is matched with whether the test was improved, hindered, or not affected.": [
                    "The improvement/hindrance of a test should be validated based on the player's traits and flaws."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
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
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional trait.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
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
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional flaw.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
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
            'max_score': 10,
            'min_score': 0
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
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who should have an additional item.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
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
            'max_score': 10,
            'min_score': 0
        }
    },
    'remove_trait': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when an existing trait should be removed from the player considering the progress and game state so far.": [
                    "If 'trait_name' does not exist in 'traits' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player whose trait should be removed.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'trait_name' is properly parsed.": [
                    "The trait name should be evaluated in terms of quality and consistency considering the progress and game state so far.",
                    "If the trait name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the trait list in the player."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'remove_flaw': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when an existing flaw should be removed from the player considering the progress and game state so far.": [
                    "If 'flaw_name' does not exist in 'flaws' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player whose flaw should be removed.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'flaw_name' is properly parsed.": [
                    "The flaw name should be evaluated in terms of quality and consistency considering the progress and game state so far.",
                    "If the flaw name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the flaw list in the player."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'remove_item': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when an existing item should be removed from the player considering the progress and game state so far.": [
                    "If 'item_name' does not exist in 'inventory' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player whose item should be removed.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'item_name' is properly parsed.": [
                    "The item name should be evaluated in terms of quality and consistency considering the progress and game state so far.",
                    "If the item name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the inventory in the player."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'use_item': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when an existing item should be used by the player considering the progress and game state so far.": [
                    "If 'item_name' does not exist in 'inventory' in the player, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player whose item should be used.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'item_name' is properly parsed.": [
                    "The item name should be evaluated in terms of quality and consistency considering the progress and game state so far.",
                    "If the item name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the inventory in the player."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_detection_of_expendability': {
            'question': "Does the function correctly detect the expendability of the item?",
            'specifications': {
                "The intermediate result of the expendability detection is matched with whether the item should be removed after the usage.": []
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'add_object': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a new object should be added to the environment considering the progress and game state so far.": [
                    "If 'object_name' already exists in 'environment' in the scene, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'object_name' is properly parsed.": [
                    "Even if the object already exists but the function has been called falsely, the argument should be considered correct if it is the right name in 'environment' in the scene.",
                    "If the object does not exist, then the name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                ],
                "'object_desc' is properly parsed.": [
                    "If the object description is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the object description is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the object name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'use_environment': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when a player tries to reach or interact with an object in 'environment' considering the progress and game state so far.": [
                    "If 'object_name' does not exist in 'environment' in the scene, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'player_name' is the name of the player who uses the object.": [
                    "This should be identical to the name of the player stored in 'players'. (including case, special symbols, etc.)",
                    "If the player name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the player list.",
                ],
                "'object_name' is properly parsed.": [
                    "The object name should be evaluated in terms of quality and consistency considering the progress and game state so far.",
                    "If the object name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the environment in the scene."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_detection_of_obtainability': {
            'question': "Does the function correctly detect the obtainability of the object?",
            'specifications': {
                "The intermediate result of the obtainability detection is matched with whether the object can be added into the player inventory.": []
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    },
    'use_random_table': {
        'correct_activation': {
            'question': "Is the function called in proper timing when it is needed?",
            'specifications': {
                "The function is called at the same turn as the user queries.": [],
                "The function is called when some entries should be sampled from a random table considering the progress and game state so far.": [
                    "If 'table_name' does not exist in 'random_tables' in the scene, this should be penalized."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_arguments': {
            'question': "Are the function arguments parsed correctly?",
            'specifications': {
                "'table_name' is properly parsed.": [
                    "The table name should be evaluated in terms of quality and consistency considering the progress and game state so far."
                    "The table should have the entries that are needed for proceeding with the game according to the scene-specific rules, game state and history so far."
                    "If the table name is somewhat correct but it is represented differently, it is considered wrong since it cannot be parsed from the random table list in the scene."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_determination_of_numbers': {
            'question': "Does the function make the right determination of the number of samples to retrieve?",
            'specifications': {
                "The intermediate result of the number of samples is matched with how many entries should be randomly sampled from the table.": [
                    "If the number is explicitly mentioned in the scene, the argument should be considered correct if it is matched with the content described in the scene.",
                    "If the number is not mentioned in the scene, the argument should be considered correct if it is natural and reasonable taking into account the table name and the current progress."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_determination_of_sample_exclusion': {
            'question': "Does the function make the right determination of whether the sampled entries should be excluded?",
            'specifications': {
                "The intermediate result of the exclusion determination is matched with whether the sampled entries should be excluded or not from the table.": [
                    "Consider whether the sampled entries should be used or set only once during the game and presenting them again later would make the game unnatural."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        },
        'correct_determination_of_table_removal': {
            'question': "Does the function make the right determination of whether the table should be removed?",
            'specifications': {
                "The intermediate result of the removal determination is matched with whether the table should be removed or not from the random table list.": [
                    "Consider whether the table should be used during the game only once during the game and using this table again later would make the game unnatural.",
                    "If there is no intermediate result of the table removal, this means the table has been automatically removed since all entries were excluded during the game. You don't have to evaluate this case."
                ]
            },
            'notes': [],
            'max_score': 10,
            'min_score': 0
        }
    }
}