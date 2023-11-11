# labyrinth-simulator-kani
### **Welcome to Labyrinth!**

<img src="figure/labyrinth.png" style="width: 40%"/>

This is an open-sourced project for simulating the fantasy text-based adventure game: **Jim Henson's Labyrinth: The Adventure Game**[[1]](#1) grounded on large language models supported by kani[[2]](#2).

The Goblin King, which is the game manager, is built on a large language model and supports the players to proceed with the game while controlling the game flow and following the game rules.

The game system is chat-based via the terminal, where the players interact with an AI working as a game manager to explore the enormous world of the Labyrinth, including talking with the NPCs, fighting against the opponents, utilizing the objects and items, and solving the interesting puzzles!

The users can have an experience on their terminal that is almost identical to actual gameplay. Note that however, this is a research-purpose project so it cannot replace the complete in-person user experience.

For the details on the game rules of Labyrinth, you can refer to [the game book](https://www.riverhorsegames.com/products/rh_lab_005). Although this repository includes the player character data, the summarized game rules, and the metadata of each game scene, it does not cover all details and contents of the tutorial book.

<br/>

This project was carried out while working as a Research Assistant at University of Pennsylvania, under the supervision of Prof. Chris Callison-Burch.

Any feedback or opinion is welcomed. Enjoy the game!

<br/>

---

### Details

There are a few technical details of the system, which can help you understand how to set the initial arguments to run the game and what are the current limitations of the project.

- **Different rule injection methods**: You can change how the model understands or leverages the game rules during the interaction.
- **Active utilization of function calling**: The game manager not only generates a natural-language response but also calls different functions depending on the need. You might experience a more flexible and interesting game flow than just a simple chat-based interaction.
- **Per-scene execution**:  You can simulate each game scene without playing the whole game from the beginning.
- **(Local) Multi-player gameplay**: The system has been implemented considering the multi-player participation.
- **Different prompt designs**: You can change how an input prompt is made for each generation. You can set the concatenation policy for combining the chat history, the number of past utterances to include, and the summarization period.
- **Flexible decoding parameters**: You can set the decoding parameters to control the output just as using the OpenAI APIs. You can refer to [the document](https://platform.openai.com/docs/api-reference/chat/create) for more details.

<br/>

---

### Limitations

<br/>

---

### Arguments

<br/>

---

### How to run

<br/>

---

<a id="1">[1]</a> Milton, B., Cæsar, J., Froud, B., & Henson, J. (2019). *Jim Henson’s labyrinth: The adventure game*. River Horse.

<a id="2">[2]</a> Zhu, Andrew, et al. "Kani: A Lightweight and Highly Hackable Framework for Building Language Model Applications." *arXiv preprint arXiv:2309.05542* (2023).
