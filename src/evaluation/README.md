# Evaluation Guideline

### Project summary

This project is for simulating the text-based adventure game, Jim Henson's Labyrinth: The Adventure game, based on the LLM-powered game manager. (a.k.a Goblin King in the game) Especially, the key of this project is the function calling, which calls a user-defined function when it is needed considering the chat messages and the game states, to perform more fine-grained controls or directly use/update the game states. 

The goal of this project is how much the function calling can improve the performance of the game manager in terms of the management of the game states and game flow.

<br/>

---

### Survey instruction

For evaluation, you will be given survey links which enable you to give a score to each target response while checking the conversation between the players and the game manager and the game states on which the current scene is grounded. We attach a few instructions for the survey to help you understand what exactly each component represents and how you can follow the survey flow.

<br/>

First, one survey contains one gameplay data played by a party consisting of 4 player characters with different profiles. They interacted with the game manager, who is called "Goblin King" in the game, to clear the game scene. You should evaluate the generated responses, each of which is set to "target response", in terms of 3 different metrics. You will be given more specific instructions for each metric on the survey page.

For each evaluation, you will be given some components you should look through, which include **Starting game state**, **Starting player state** of each player, **Chat history so far**, and **Target response**. We introduce the details of these as follows:

<br/>

#### 1. Starting game state

The starting game state shows the initial game scene when the scene has started. Note that this is the starting state, so while proceeding with the game, the state might have been updated, which you should also consider for your evaluations. (The starting state does not contain the updates!)

- <u>**Chapter**</u>: The chapter name.
- <u>**Scene**</u>: The scene name.
- <u>**Scene Summary**</u>: Overall description of the current scene.
- <u>**NPCs**</u>: Initialized NPCs. The name of each NPC is tagged with **[Name]**. Below it, you can check the specifications of that NPC:
  - **Kin**: The kin of the NPC.
  - **Persona**: The persona of the NPC.
  - **Goal**: The goal of the NPC.
  - **Trait**: The trait of the NPC which would be helpful. (Trait name - Trait description)
  - **Flaw**: The flaw of the NPC which would be harmful. (Flaw name - Flaw description)
- <u>**Success Condition**</u>: The condition for the players to win this scene.
- <u>**Failure Condition**</u>: The condition for the players to lose this scene.
- <u>**Game Flow**</u>: The intended game flow of the current game scene. This might not have necessarily to be followed, but gives a useful hint to the game manager which is what should be done next during the game.
- <u>**Environment**</u>: The environmental objects. The name of each object is tagged with **[Name]**. Next to it, you can check the description of that object.
- <u>**Random Tables**</u>: The random tables. The name of each table is tagged with **[Name]**. Below it, you can check the actual entries of that table.
- <u>**Consequences**</u>: The consequences after finishing the scene.
- <u>**Action Scene**</u>: Indication of whether the action scene is currently activated or not. (True/False)

<br/>

#### 2. Starting player state

Starting player state shows the initialized state of each player character when the scene has started. Like the starting game state, these player states might have also been updated. So you should follow the dialogue and keep in mind these updates during the game for a correct evaluation. The name of each player is shown as **Starting player state of ...**

- <u>**Kin**</u>: The kin of the player.
- <u>**Persona**</u>: The persona of the player.
- <u>**Goal**</u>: The goal of the player.
- <u>**Traits**</u>: The traits of the player. The name of each trait is tagged with **[Name]**. Next to it, you can check the description of that trait.
- <u>**Flaws**</u>: The flaws of the player. The name of each flaw is tagged with **[Name]**. Next to it, you can check the description of that flaw.
- <u>**Inventory**</u>: The inventory of the player. The name of each item is tagged with **[Name]**. Next to it, you can check the description of that item.
- <u>**Additional Notes**</u>: The additional notes for the player. This contains any specific behavior that the game manager should take when the PC does something. (e.g. Adding a flaw when a firey detaches its body part.)

<br/>

#### 3. Chat history so far

The chat history which has been done between the players and the game manager can be seen in this part. The message format is **"[Role] Name: Content"**. If the role is "Player", the name is supposed to the name of the player. If the role is "Game Manager", this is a message from the Goblin King. Note that the Goblin King can also generate multiple messages at a time, so a message by the game manager can come right after another message by the game manager.

 <br/>

#### 4. Target response

This is the response you should actually evaluate. It has been generated by the game manager given the game states and the chat history. The format is the same as the one explained in **3. Chat history so far**.


<br/>

---

### Additional notes

1. While we assume that you are aware of the general rules of the Labyrinth game, we attach the rule summary here to make sure that we are on the same page. This is a summarized version, which might miss some details in the actual game book. But there would be no problem if you just fully understand the game rules here since all agents in this game have worked under this summarization.
   - **Player characters**
     - Each player has their own character. Basically, each character has its name, kin, persona, and goal. Also, each player character has their own traits, flaws, and items. The Goblin King should consider these properties of players when he interacts with them and the players should try to be consistent with their characters while proceeding with the game.
   - **Difficulty**
     - The player party will come across many challenges and when a player character tries something that has a chance of failure, there should be a test by rolling a dice. The Goblin King should decide how difficult the test is by choosing a number between 2 and 6. Then, the Goblin King should let the player roll a die and notify if the result is equal to or higher than the difficulty number. If it is, the player succeeds in passing, otherwise the player fails. The result 1 is always considered as failure.
     - If the player has a trait which can improve the test, the player rolls the dice twice and takes the larger value. If the player has a flaw which can hinder the test, the player rolls the dice twice and takes the smaller value.
     - If other players declare to help the test of a player, they should specify how they can help with the traits they have. Each player that is helping reduces the difficulty by 1 to a minimum of 2.
   
   - **Equipment**
     - The players might find items or equipment throughout the Labyrinth. Players can ask the Goblin King if they can use the items, and the Goblin King may decide if an item improves a test or allows the player to use it without rolling a dice.
     - The maximum number of items a player can carry is 6. If a player wants to pick up an additional item but the inventory is full, an item from the inventory should be dropped.
   - **Action scenes**
     - An action scene is a situation where the players should do something under a strict time limit. If they do nothing, the situation will change drastically. 
     - An action scene is declared by the Goblin King. After the action scene is activated, each player should describe one action to do per player. Then the Goblin King asks the player to test if the action is successful via dice roll. Each round during the action scene is about 5 seconds long, so the players should be hurry to take some actions. After each player has made the tests, the Goblin King should decide how the scene has gone as an outcome. If there are reactions from foes or other objects, the Goblin King can ask the players for other actions and tests to avoid the reaction. The Goblin King should keep describing the changes after the players' action and requiring them to take other actions as a counter. 
     - When the Goblin King thinks that the action scene has finally ended, this should be announced to the players to terminate the action scene. If the Goblin King is confused with whether it is an action scene or a simple test, one consideration is if there are multiple possible outcomes. If there are, declaring an action scene might be better, otherwise just a test.
   - **Using tables**
     - Using tables: The random tables are used during the scene if there are some objects that should be brought in or some entries should be selected for proceeding the scene. The Goblin King should first determine if the current game flow requires to use the random table, decide which table should be referred to, pick the entries depending on the requirement, and use them during the game. The sampled entries can be used for just adding the context during the gameplay, or directly updating the required game states.
   - **Hours**
     - Other than the time limit during an action scene, the Labyrinth has the total time limit per each scene. The Goblin King should count the time and tell the players everytime they lose an hour. If 13 hours has passed, the players lose the game regardless of the current game state.
   - **NPCs**
     - There are various NPCs in the Labyrinth who the player party can interact with or confront. If the players try to chat with an NPC, the Goblin King should act out the NPC's part by finding the corresponding NPC's name in the scene and checking its specifications. The Goblin King might have to mimic its tone, characteristics, or behaviors genuinely. The NPC's line is tagged with additional ". 
     - If there is no name which is matched with the one the players mentioned, a new NPC might have to be generated and initialized in the scene. However, if the new NPC doesn't fit the given scene or context so far, the Goblin King should reject it and tell the players the reason. 
     - Sometimes, the players can make friends with NPCs by convincing them during the game. However, the NPCs are usually afraid of the Goblin King, they would not stay long. Depending upon an NPC's flaw, they may leave the group at certain times depending on it's flaw. To see if an NPC is leaving, make the players roll a test with difficulty 4. If the test is successful, the NPC stays in the party. If the Goblin King himself appears in the scene, all NPCs will flee without any tests.
   - **Additional notes**
     - Each participant must adhere to their respective roles, which means that the Goblin King should keep his role as the game manager and the players should keep their roles as the players. The Goblin King should keep track of the game flow and try to lead the game which should not be too far from the intended flow and the scene. If the Goblin King thinks that the current circumstance aligns with the success/failure condition, he should announce it so that the game can be correctly terminated.
   
2. For technical reasons, there might some exceptions you should consider for your evaluations.
   - **The time limit for the game and action scenes are not included in the gameplay data.** You just have to check whether the action scene has been activated/terminated according to the dialogue, not checking whether the Goblin King has actually imposed a strict time limit for each player. Also, you don't have to validate whether the game manager notifies the players when they lose an hour.
   - **The player might act like the game manager, such as describing the scene or talking as an NPC.** The player is also an AI agent and sometimes does these undesired things. Since you are not evaluating the players, you don't have to penalize these unexpected behaviors of the players. <u>However, the game manager should be penalized if it does nothing about these player's undesired violations.</u> (Details on the reliability rubrics)
   - **The dialogue might go far beyond the intended game flow.** While the game manager has the intended game flow in the game states, sometimes the game gets weird or repetitive depending on the dialogue. This is usually not acceptable, but if you think the game manager tries to resolve this and get the game back on the right track, that would not be a problem. Sometimes, the dialogue can fall into an infinite loop. This is caused by the failure to terminate the game at the proper time or by a infinite cycle where the players and game manager wait for the dice roll results. The termination issue is usually not the fault of the game manager, so if the game manager tries to end the conversation or resolve this unexpected loop on its own, that's fine. <u>However, if it makes up the game scene or improvises something which is entirely unrelated to the game scene during this infinite loop, this should be penalized.</u> (Details on the reliability rubrics) On the other hand, if the loop is because of the dice roll, this IS the fault of the game manager because the game manager should notify (or at least improvise) the dice roll result. <u>Therefore, for in case of the dice roll waiting, the response should be penalized.</u>

<br/>

---

Thank you for all your help!