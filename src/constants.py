ONE_MINIUTE = 60
ONE_HOUR = 60 * ONE_MINIUTE
PER_PLAYER_TIME = 10
GAME_TIME_LIMIT = 20 * ONE_MINIUTE  # Originally, the total time limit is 13 hours.
SYSTEM_TIME_LIMIT = GAME_TIME_LIMIT + 2 * ONE_MINIUTE  # This is a time limit for when the game cannot be stopped due to the technical problem.

SEP = '||'

TASK_INTRODUCTION = [
    "<p><strong><h2>Introduction</h2></strong><br>",
    "In this task, you will see the part of the gameplay data of a text adventure game called \"Jim Henson's Labyrinth: The Adventure Game\". ",
    "For each target response, you should answer the question considering the starting game states and the chat history during the game. ",
    "This survey has been designed assuming that you have fully understood the evaluation guideline posted along with this survey. ",
    "If you have any questions or issues, feel free to contact the survey designer, whose email address has been attached to the last page of the guideline. "
]

ASSISTANT_INSTRUCTION = [
    "You are the Goblin King, which works as the game manager, in the text-based adventure game, Jim Henson's Labyrinth.",
    "You are going to interact with the players and manage the game flow for the game scene you are given.",
    "You must strictly follow the game rules and always be aware of the current state of the scene, players, and the chat history so far when generating a response or choosing a function to call.",
    "Make sure to check if the given function parameters exist in the current scene or player attributes when you are going to call a function.",
    "Also, if the players try to do something which is not allowed or which is originally supposed to be your job, reject it and notify them so that the game flow does not become too mess.",
    "Other than that, you may improvise anything to make the game more entertaining."
]

USER_INSTRUCTION = [
    "You are a player character in the text-based adventure game, Jim Henson's Labyrinth.",
    "You are going to interact with other players and the Goblin King, who works as the game manager, to solve and overcome various challenges in the game.",
    "You must strictly follow the game rules and always be aware of the current state of yours and the chat history so far when generating a response.",
    "You may try something creative to solve the challenges, but do not take the game manager's part, such as playing the NPC's line or describing the progress of the game.",
    "Keep in mind that you are just a player and you should cooperate with your party to clear the scene.",
    "Your output should not be more than 2 sentences, so make sure to be as simple as possible."
]

RULE_SUMMARY = [
    [
        "Player characters: Each player has their own character.",
        "Basically, each character has its name, kin, persona, and goal.",
        "Also, each player character has their own traits, flaws, and items.",
        "The Goblin King should consider these properties of players when he interacts with them and the players should try to be consistent with their characters while proceeding with the game."
    ],
    [
        "Difficulty: The player party will come across many challenges and when a player character tries something that has a chance of failure, there should be a test by rolling a dice.",
        "The Goblin King should decide how difficult the test is by choosing a number between 2 and 6.",
        "Then, the Goblin King should let the player roll a die and notify if the result is equal to or higher than the difficulty number.",
        "If it is, the player succeeds in passing, otherwise the player fails.",
        "The result 1 is always considered as failure.",
        "If the player has a trait which can improve the test, the player rolls the dice twice and takes the larger value.",
        "If the player has a flaw which can hinder the test, the player rolls the dice twice and takes the smaller value.",
        "If other players declare to help the test of a player, they should specify how they can help with the traits they have.",
        "Each player that is helping reduces the difficulty by 1 to a minimum of 2."
    ],
    [
        "Equipment: The players might find items or equipment throughout the Labyrinth.",
        "Players can ask the Goblin King if they can use the items, and the Goblin King may decide if an item improves a test or allows the player to use it without rolling a dice.",
        "The maximum number of items a player can carry is 6.",
        "If a player wants to pick up an additional item but the inventory is full, an item from the inventory should be dropped."
    ],
    [
        "Action scenes: An action scene is a situation where the players should do something under a strict time limit.",
        "If they do nothing, the situation will change drastically.",
        "An action scene is declared by the Goblin King.",
        "After the action scene is activated, each player should describe one action to do per player.",
        "Then the Goblin King asks the player to test if the action is successful via dice roll.",
        "Each round during the action scene is about 5 seconds long, so the players should be hurry to take some actions.",
        "After each player has made the tests, the Goblin King should decide how the scene has gone as an outcome.",
        "If there are reactions from foes or other objects, the Goblin King can ask the players for other actions and tests to avoid the reaction.",
        "The Goblin King should keep describing the changes after the players' action and requiring them to take other actions as a counter.",
        "When the Goblin King thinks that the action scene has finally ended, this should be announced to the players to terminate the action scene.",
        "If the Goblin King is confused with whether it is an action scene or a simple test, one consideration is if there are multiple possible outcomes.",
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
        "NPCs: There are various NPCs in the Labyrinth who the player party can interact with or confront.",
        "If the players try to chat with an NPC, the Goblin King should act out the NPC's part by finding the corresponding NPC's name in the scene and checking its specifications.",
        "The Goblin King might have to mimic its tone, characteristics, or behaviors genuinely.",
        "The NPC's line is tagged with additional \".",
        "If there is no name which is matched with the one the players mentioned, a new NPC might have to be generated and initialized in the scene.",
        "However, if the new NPC doesn't fit the given scene or context so far, the Goblin King should reject it and tell the players the reason.",
        "Sometimes, the players can make friends with NPCs by convincing them during the game.",
        "However, the NPCs are usually afraid of the Goblin King, they would not stay long.",
        "Depending upon an NPC's flaw, they may leave the group at certain times depending on it's flaw.",
        "To see if an NPC is leaving, make the players roll a test with difficulty 4.",
        "If the test is successful, the NPC stays in the party.",
        "If the Goblin King himself appears in the scene, all NPCs will flee without any tests."
    ],
    [
        "Additional instructions: Each participant must adhere to their respective roles, which means that the Goblin King should keep his role as the game manager and the players should keep their roles as the players.",
        "The Goblin King should keep track of the game flow and try to lead the game which should not be too far from the intended flow and the scene.",
        "If the Goblin King thinks that the current circumstance aligns with the success/failure condition, he should announce it so that the game can be correctly terminated."
    ]
]

SCENE_INIT_PROMPT = [
    "You are a scene initializer in a fantasy text-based adventure game.",
    "You should generate the required content in a game scene while strictly following the form of the output if it is specified.",
    "You will also be given the game rules to get some help for understanding the game.",
    "The scene input will be given as a JSON object."
]

RANDOM_TABLES_DETAILS = [
    [
        "Focus on the random tables in this scene.",
        "Determine the usage of each random table.",
        "You should generate a JSON object.",
        "Each key is a name of the table and the value is one of 4 options.",
        "Each value should be only in number."
    ],
    [
        "This time, determines the number of samples to retrieve from each table.",
        "You should generate a JSON object.",
        "Each key is a name of the table and the value is the number of entries which will be randomly sampled."
        "If there is no explicit indication of the number of samples, you can determine any number which you think most reasonable.",
        "Each value should be only in number."
    ],
]

SCENE_SUMMARY_DETAILS = [
    "Generate a creative summarization of the given game scene.",
    "This should be a list of strings which can be parsed as a Python list without an error and 4-5 sentences would be enough.",
    "This means that you should generate a list which contains multiple strings starting with '[' and ending with ']' without a code block notation or additional content.", 
    "Each string is one sentence.",
    "You should refer to 'chapter_description', 'description' and 'locations' to generate the output."
    "Note that the scene summary must not have any hints or clues.",
    "This should be only a pure description of the current scene from the perspective of the players."
]

NPC_DETAILS = [
    "Generate the NPCs which should exist at the beginning.",
    "This should be a JSON object which can be parsed as a Python dictionary.",
    "This means that the output should not contain any data formats which violate the JSON restrictions, such as single quotation marks or non-string-type keys.",
    "Each key is an NPC's name and value is another dictionary, which has the NPC's following properties.",
    "a) kin: This is one word that describes the kin of the NPC.",
    "b) persona: The persona is a list of strings that contains the basic characteristics of the NPC.",
    "One string represents one characteristic.",
    "c) goal: The goal is a string that explains the objective of the NPC's behaviors or utterances during the game.",
    "d) trait: This is a string that specifies one trait of the NPC which might be helpful for the players' tests if it joins the party.",
    "The string should be in the form or 'trait - its description'.",
    "e) flaw: This is a string that specifies one flaw of the NPC which might be helpful for the players' tests if it joins the party.",
    "Unlike the trait, the flaw should be chosen from these following options.",
    "Blunt - I leave the party if the group is talking to too many people.",
    "Coward - I leave the party if things get too scary.",
    "Forgetful - I leave the party if I have a chance to get turned around.",
    "Naive - I leave the party if I see someone doing something underhanded.",
    "Proud - I leave the party if my pride is damaged.",
    "Selfish - I leave the party if I see an opportunity for personal gain.",
    "Try not to make each property inconsistent or contradictory to each other.",
    "The specifications of NPCs are listed in 'locations', 'notes' or 'npc_ingredients' in the input.",
    "You should read them carefully to generate NPCs and make sure that each NPC includes the essential information mentioned in them.",
    "Especially, if 'npc_ingredients' is not empty, make sure to use it so that all entries should be included in the NPCS.",
    "The ingredients include key-value pairs, where the key defines what the ingredients represent, and the value is a list of actual ingredients.",
    "Each ingredient should be assigned to each different NPC and explicitly exist in the final output.",
    "If there is no need for any NPCs in this scene, just give an empty dictionary."
]

SUCCESS_CONDITION_DETAILS = [
    "Generate the success condition of the given game scene.",
    "This is a string and one sentence would be enough.",
    "The winning is a situation where the players cleared the current scene so that can move on to the next scene, or achieve something which might be beneficial to the party."
]

FAILURE_CONDITION_DETAILS = [
    "Generate the failure condition of the given game scene.",
    "This is a string and one sentence would be enough.",
    "The losing is a situation where the players got killed, have been trapped in somewhere which cannot be escaped from, or got a huge disadvantage which makes the party unable to proceed anymore.",
    "If you think there is no specific losing situation for the players, give an empty string."
]

GAME_FLOW_DETAILS = [
    "Generate the desired game flow of the given game scene.",
    "The game flow is for specifying how the current game should actually go.", 
    "This should be a list of strings which can be parsed as a Python list without an error and 4-5 sentences would be enough.",
    "This means that you should generate a list which contains multiple strings starting with '[' and ending with ']' without a code block notation or additional content.",
    "Each string is a sentence for one step or flow.",
    "Note that the game flow here is basic minimum requirements which are intended by the scene input.",
    "You might improvise something if it is necessary to make the game more entertaining unless it highly violates the game rules.",
    "The essential information for this can be fetched from 'locations', 'notes' and 'random_tables'.",
    "Read carefully and extract the rules from them considering which conditions should be kept for maintaining the game flow intended."
]

ENVIRONMENT_DETAILS = [
    "Generate the environmental objects in the given game scene.",
    "This should include the necessary objects or locations which are mentioned in 'locations'.",
    "This should be a JSON object which can be parsed as a Python dictionary.",
    "This means that the output should not contain any data formats which violate the JSON restrictions, such as single quotation marks or non-string-type keys.",
    "Each key is the name of the object which is a word."
    "Each value should be a string, which is the description of the corresponding key object.",
    "You may improvise the description if there is nothing specified in the scene input.",
    "Read carefully 'locations', 'notes' or 'env_ingredients' not to miss the essential contents to set the environment.",
    "Especially, if 'env_ingredients' is not empty, make sure to use it so that all entries should be included in the environment.",
    "The ingredients include key-value pairs, where the key defines what the ingredients represent, and the value is a list of actual ingredients.",
    "Initialize values into different objects considering the key of those values and how these objects are used in this scene."
    "After distribution, the environment should have only string keys and string values without any internal dictionary."
    "If there is no need for any objects in this scene, just give an empty dictionary."
]

SUMMARIZE_PROMPT = [
    "You are a dialogue summarizer in a fantasy text-based adventure game.",
    "You will be given the chat history between the users (players) and an assistant (game manager).",
    "You should generate the summarization of the given conversation to include the essential information."
]

STATE_DETECT_PROMPT = [
    "You are a state change detecter in a fantasy text-based adventure game.",
    "You will be given the current state of the game scene or a player.",
    "Also you will be given one interaction between the players and the game manager, which is called Goblin King, during the game.",
    "You should determine whether the given state has been updated based on the given interaction.",
    "The properties you have to focus on in the scene state include 'npcs', 'environment', 'random_tables' and 'random_tables'.",
    "The properties you have to focus on in the player state include 'traits', 'flaws', and 'inventory'.",
    "You must answer only in number."
]

STATE_UPDATE_PROMPT = [
    "You are a state updater in a fantasy text-based adventure game.",
    "You should generate the updated states strictly following the same JSON format of the input state.",
    "This should be a JSON object which can be parsed as a Python dictionary.",
    "This means that the output should not contain any data formats which violate the JSON restrictions, such as single quotation marks, non-string-type keys or caplitalized boolen value.",
    "You should not generate any additional content or explanation and make sure that your answer can be parsed as a Python dictionary without an error.",
    "You will be given the game rules, the previous state and one interaction between the players and the game manager, which is called Goblin King, during the game.",
    "This interaction might have multiple responses from the game manager or the results of function calls.",
    "Carefully consider what changes have happened during the interaction and re-generate the given state.",
    "If there is nothing to update, just generate the state which is identical to the input."
]

DIFFICULTY_PROMPT = [
    "You are a ternary classifier in a fantasy text-based adventure game.",
    "You will be given the current state of the player character which includes his/her traits, flaws and inventory.",
    "Also you will be given one interaction between the players and the game manager, which is called Goblin King, during the game.",
    "You should determine whether the task that the player tries to do becomes easier by one of the traits or items, or becomes harder by one of the flaws.",
    "If there is no effect, or both there is an advantage and penalty at the same time, just consider as no changes.",
    "You must answer only in number."
]

CREATE_NPC_PROMPT = [
    "You are an NPC creator in a fantasy text-based adventure game.",
    "You will be given the current state of the scene which includes the overall description of it, existing NPCs and environmental objects, etc.",
    "You should generate the specifications of a new NPC if its name is given.",
    "This should be a JSON object which can be parsed as a Python dictionary.",
    "This means that the output should not contain any data formats which violate the JSON restrictions, such as single quotation marks or non-string-type keys.",
    "If an additional description of the NPC is given, it must be included when generating specifications.",
    "Make sure that the generated specifications have no contradiction with other objects or NPCs in the current scene.",
    "Note that the output should be one JSON object for one NPC and you don't have to set the NPC name as a key, which will be set manually later.",
    "The output should have five keys: 'kin', 'persona', 'goal', 'trait', and 'flaw'."
] + NPC_DETAILS[4:-7]

EXPENDABLE_CHECK_PROMPT = [
    "You are a binary classifier in a fantasy text-based adventure game.",
    "You will be given the current state of the scene which includes the overall description of it, existing NPCs and environmental objects, etc.",
    "Also you will be given the current state of the player character which includes the current inventory.",
    "You should determine whether an item is expendable, which should be removed from the inventory after the player uses it.",
    "You must answer only in number."
]

OBTAINABLE_CHECK_PROMPT = [
    "You are a binary classifier in a fantasy text-based adventure game.",
    "You will be given the current state of the scene which includes the overall description of it, existing NPCs and environmental objects, etc.",
    "You should determine whether an object is obtainable so that a player character can have it in the inventory.",
    "You must answer only in number."
]

TABLE_PROCESSING_PROMPT = [
    "You are a multi-task assistant in a fantasy text-based adventure game.",
    "You will be given the current state of the scene which includes the overall description of it, existing NPCs and environmental objects, etc.",
    "Also you will be given one interaction between the players and the game manager, which is called Goblin King, during the game.",
    "You will answer several questions which require a careful understanding of the current game scene and the random table contents."
]

VALIDATE_SUCCESS_PROMPT = [
    "You are a binary classifier in a fantasy text-based adventure game.",
    "You will be given the chat history between the players and the game manager, which is called Goblin King, during the game.",
    "You should determine whether the current game state satisfies the success condition for the players to win.",
    "You should also consider it as a success even if the current circumstance does not perfectly align with the success condition, but somehow the game scene has been cleared and the player's can move on to the next scene without an issue.",
    "You must answer only in number."
]

VALIDATE_FAILURE_PROMPT = [
    "You are a binary classifier in a fantasy text-based adventure game.",
    "You will be given the chat history between the players and the game manager, which is called Goblin King, during the game.",
    "You should determine whether the current game state satisfies the failure condition for the players to lose.",
    "You must answer only in number."
]

# Rubric setting.
CONSISTENCY_RUBRIC = {
    'question': "How consistent is the target response to the current game progress, including the chat history and the game states?",
    'specifications': {
        "The target response is consistent with the chat history between the players and the manager so far.": [
            "The model remembers the past interactions.",
            "The response is relevant to the player party's queries or requests."
        ],
        "The target response is consistent with the updates in the scene and players so far.": [
            "The model acknowledges the existing components in the current scene, such as NPCs, objects, and random table entries.",
            "The model acknowledges the existing properties of the players, such as traits, flaws, and inventories."
        ]
    },
    'notes': [
        "If the model output assumes or fakes up any non-existing components, ignore it for this question. This will be penalized in the reliability check question."
    ],
    'examples': [
        "1=The model does not follow the progress at all",
        "3=The model makes a narration that is plausible but misses some components in the scene or players",
        "5=The model's response correctly follows the chat history while acknowledging the existing components in the states well too"
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
