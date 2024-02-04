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
- **Different prompt designs**: You can change how an input prompt is made for each generation. You can set the concatenation policy for combining the chat history, the number of past utterances to include, and the summarization period.
- **Supporting performance evaluation**: The project supports a separate script for evaluating the model performance in each sub-task. Also, you can export the whole game state and history for further human evaluation.
  - Currently, the evaluation script supports the performance validations for: **1) Scene Initialization** and **2) Rule Understanding**.


<br/>

---

### Arguments

**Arguments for the gameplay**

| Argument             | Type           | Description                                                  | Default           |
| -------------------- | -------------- | ------------------------------------------------------------ | ----------------- |
| `--seed`             | `int`          | The random seed for randomized operations.                   | `0`               |
| `--model_idx`        | `str`          | The index of the model. Since only `openai` engine is supported for leveraging the function calling feature, the model should be the one from OpenAI API. Check kani's doc (https://kani.readthedocs.io/en/latest/engine_reference.html#)[https://kani.readthedocs.io/en/latest/engine_reference.html#] to see the available models for this argument. | `gpt-4`           |
| `--rule_injection`   | `str`          | The rule injection policy. The available options include: 1) `None` - We don't inject any rule. This tests the default knowledge in the pre-trained model. 2) `full` - The summarized game rules are always included in the system prompt. The summarization is stored in `src/constants.py`. 3)`retrieval` - The system fetches the relevant rule segments every time the model generates a response. | `full`            |
| `--scene_idx`        | `int`          | The index of the scene to play. Note that you should specify the correct index of the scene list, which is stored in`data/scenes.json`. | `0`               |
| `--num_players`      | `int`          | The number of players.                                       | `1`               |
| `--init_scene`       | `store_true`   | Setting whether to newly initialize the scene or re-use the pre-initialized scene. The initialized scene will be stored in `initialized/{MODEL_IDX}-{SCENE_IDX}.json`. It this argument is not set, but there is no initialized file, the scene will be initialized by default. | -                 |
| `--export_data`      | `'store_true'` | Setting whether to export the gameplay data after the game for the evaluation purpose. The exported result will be stored in `results/{YOUR_ID}-{TIME}.json`. | *Set by default.* |
| `--automated_player` | `'store_true'` | Setting another kanis for the players for simulating the game automatically. | -                 |

<br/>

**Arguments for the prompt construction**

| Argument           | Type           | Description                                                  | Default  |
| ------------------ | -------------- | ------------------------------------------------------------ | -------- |
| `--concat_policy`  | `str`          | The concatenation policy for including the previous chat logs. The available options include: 1) `simple` - The manager simply concatenates the most recent turns. 2) `retrieval` - The manager retrieves the most relevant utterances from the history using sentence embedding and cosine similarity. Note that the current user inputs are always included. | `simple` |
| `--max_turns`      | `int`          | The maximum number of turns to be included. If it is not specified, the model includes as many turns as possible. Note that without this argument, the retrieval method for concatenation will work identically to the simple concatenation. | -        |
| `--summarization`  | `'store_true'` | Setting whether to include the summarization or not. The system will summarize the chat logs when a certain number of turns has reached(`--summ_period`), and add the output to the chat history. The summarized logs are also considered as the chat logs and fetched according to `--concat_policy` and `--max_turns`. | -        |
| `--summ_period`    | `int`          | The summarization period in terms of the number of turns. If a value $p$ is set for this argument, the system will summarize the last $p$ turns when the number of logs becomes a multiple of $p$. Note that if this is not specified but only `--summarization` is set, the system will ignore `--concat_policy` and `--max_turns` and summarize as many logs as possible to make a prompt only with the summarization and current queries. (This is definitely different from setting `--summ_period=1`!) | -        |
| `--clear_raw_logs` | `store_true`   | Setting whether to remove the raw chat logs after the summarization. That is, except for the turns which have not been summarized yet, the rest of the logs included are all summarized logs. | -        |

<br/>

**Arguments for the response generation**

Note that these are only used for the actual interaction during the game. Other tasks, such as initializing a scene, classification-based decisions in the functions, and summarization, will have default decoding parameters. You can refer to [the document](https://platform.openai.com/docs/api-reference/chat/create) for more details.

| Argument              | Type    | Description                                                  | Default |
| --------------------- | ------- | ------------------------------------------------------------ | ------- |
| `--max_tokens`        | `int`   | The maximum number of tokens to generate.                    | -       |
| `--frequency_penalty` | `float` | A positive value penalizes the repetitive new tokens. (-2.0 - 2.0) | `0.5`   |
| `--presence_penalty`  | `float` | A positive value penalizes the new tokens based on whether they appear in the text so far. (-2.0 - 2.0) | `0.5`   |
| `--temperature`       | `float` | A higher value makes the output more random. (0.0 - 2.0)     | `1.0`   |
| `--top_p`             | `float` | The probability mass which will be considered for the nucleus sampling. (0.0 - 1.0) | `0.8`   |

<br/>

**Arguments for the evaluation**

These are for using the separate evaluation script to test each individual model's capability on different tasks. The user can manually check the model's response and give a score for each task/question.

| Argument           | Type  | Description                                                  | Default  |
| ------------------ | ----- | ------------------------------------------------------------ | -------- |
| `--eval_name`      | `str` | The name of the evaluation task. The currently available options include: 1) `init` - The scene initialization for a given scene input. 2) `rules` - The understanding of the game rules based on Q&A form. | `init`   |
| `--engine_name`    | `str` | The name of the engine for running kani corresponding to the language model used. Check kani's doc ([https://kani.readthedocs.io/en/latest/engines.html](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkani.readthedocs.io%2Fen%2Flatest%2Fengines.html)) to see the available options for this argument. (Currently, only `openai` is supported.) | `openai` |
| `--model_idx`      | `str` | The index of the model.                                      | `gpt-4`  |
| `--rule_injection` | `str` | The rule injection policy. The available options include: 1) `None` - We don't inject any rule. This tests the default knowledge in the pre-trained model. 2) `full` - The summarized game rules are always included in the system prompt. The summarization is stored in `src/constants.py`. 3)`retrieval` - The system fetches the relevant rule segments every time the model generates a response. Note that for the evaluation `init`, the model always uses `full` injection no matter which value is set for this argument. | `full`   |
| `--scene_idx`      | `int` | The index of the scene for the initialization evaluation. Note that you should specify the correct index of the scene list, which is stored in`data/scenes.json`. Note that this does not used for the evaluation `rules`. | `0`      |

<br/>

---

### How to run

1. In your virtual environment, install the required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Modify the arguments in `exec_main.sh` to run a game scene with your preferences.

   <br/>

3. Enjoy the game!

   ```shell
   sh exec_main.sh
   ```

<br/>

For running the evaluation script, run the command below after modifying the arguments in `exec_evaluate.sh`.

```shell
sh exec_evaluate.sh
```

<br/>

---

### Limitations & Future improvements

- **Per-scene execution**: The system is run on scene-by-scene. Currently, this project does not support the whole game. In other words, while the user can run each scene for simulation, but the state or result from the completion of a scene is not stored and not connected to the next scene execution.
- **Local multi-player gameplay**: While the system has been implemented considering the multi-player participation, the project does not support remote gameplay. That means the users should play the game on one machine.

<br/>

---

<a id="1">[1]</a> Milton, B., Cæsar, J., Froud, B., & Henson, J. (2019). *Jim Henson’s labyrinth: The adventure game*. River Horse.

<a id="2">[2]</a> Zhu, Andrew, et al. "Kani: A Lightweight and Highly Hackable Framework for Building Language Model Applications." *arXiv preprint arXiv:2309.05542* (2023).
