# Evaluation Guideline

### Project summary

This project is for simulating the text-based adventure game, Jim Henson's Labyrinth: The Adventure game, based on the LLM-powered game manager. (a.k.a Goblin King in the game) Especially, the key of this project is the function calling, which calls a user-defined function when it is needed considering the chat messages and the game states, to perform more fine-grained controls or directly use/update the game states. 

The goal of this project is how much the function calling can improve the performance of the game manager in terms of the management of the game states and game flow.

You can find the details of the project [here](https://docs.google.com/presentation/d/1cKqnnXSyjdAapoRutGWBpGQxWYKchQKYRBQcqCg4Wv4/edit?usp=sharing) (A little bit outdated, but it will still help you to understand the overall idea of the project.)

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



<br/>

---

**If you have any issues or questions, feel free to contact me: jwsong05@seas.upenn.edu.**

Thank you for your all hard works!