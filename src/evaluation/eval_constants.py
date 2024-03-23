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
MAIN_OPTIONS = [
    "Check the current scene state.",
    "Check the current player states",
    "Check the past chat history.",
    "Check the current queries.",
    "Check the game rules.",
    "Give score."
]

# Rubric setting.
STATE_CONSISTENCY_RUBRIC = {
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