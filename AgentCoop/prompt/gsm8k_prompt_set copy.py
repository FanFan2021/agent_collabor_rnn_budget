from typing import Dict, Any
import itertools
from AgentCoop.prompt.prompt_set import PromptSet
from AgentCoop.prompt.prompt_set_registry import PromptSetRegistry
from AgentCoop.prompt.common import get_combine_materials
import re
from typing import Union, List
roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                        #  'Programming Expert',
                         'Inspector',
                         'Heuristic Strategist',
                         'Formal Logician',])

ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The answer should be an integer."
        "The last line of your output contains only the final result without any units, put the ONLY INTEGER final answer inside \\boxed{ONLY_INTEGER_FINAL_ANSWER}, for example: The answer is \\boxed{140}\n",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The answer should be an integer."
        "The last line of your output contains only the final result without any units, put the final answer inside \\boxed{INTEGER_FINAL_ANSWER}, for example: The answer is \\boxed{140}\n"
        "You will be given some examples you may refer to.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to.",

    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The answer should be an integer."
        "The last line of your output contains only the final result without any units, put the ONLY INTEGER final answer inside \\boxed{ONLY_INTEGER_FINAL_ANSWER}, for example: The answer is \\boxed{140}\n"
        "You will be given some examples you may refer to.",
    "Heuristic Strategist":
        "You are a Heuristic Strategist. Before performing any calculations, your task is to decompose the complex math problem into a sequence of smaller, manageable sub-problems. "
        "1. Identify the core mathematical goal and the intermediate milestones required to reach it. "
        "2. Outline the 'strategy' (e.g., 'First find the total cost, then calculate the discount, then divide by the number of people'). "
        "3. Execute your plan step-by-step to arrive at the solution. "
        "The answer should be an integer."
        "The last line of your output contains only the final result without any units, put the ONLY INTEGER final answer inside \\boxed{ONLY_INTEGER_FINAL_ANSWER}.for example: The answer is \\boxed{140}\n\n",
    "Formal Logician":
        "You are a Formal Logician. Your goal is to eliminate linguistic ambiguity by translating the word problem into formal algebraic equations. "
        "1. Assign variables (e.g., x, y, z) to the unknown quantities. "
        "2. Translate each sentence of the problem into a mathematical equation. "
        "3. Solve the resulting system of equations using algebraic manipulation. "
        "Do not rely on common-sense guesses; rely strictly on the derived equations. "
        "The answer should be an integer."
        "The last line of your output contains only the final result without any units, put the ONLY INTEGER final answer inside \\boxed{ONLY_INTEGER_FINAL_ANSWER}.for example: The answer is \\boxed{140}\n\n"
}

ROLE_CONNECTION = [
    ('Mathematical Analyst', 'Math Solver'),
    ('Mathematical Analyst', 'Programming Expert'),
    ('Mathematical Analyst', 'Inspector'),
    ('Math Solver', 'Programming Expert'),
    ('Programming Expert', 'Math Solver'),
    ('Programming Expert', 'Inspector'),
    ('Inspector', 'Math Solver'),
    ('Inspector', 'Programming Expert'),
    ('Inspector', 'Mathematical Analyst'),
]


# This function is inspired by/derived from the implementation in the following GitHub repository:
# Repository: https://github.com/chuanyang-Zheng/Progressive-Hint/blob/main/prompt/complex/complex_PHP_gsm8k.txt
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/tora/gsm8k.md
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/cot/gsm8k.md
FEW_SHOT_DATA = {
"Math Solver":
"""
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. 
They have 2 chapters of their textbook to study and 4 worksheets to memorize. 
They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. 
If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, 
include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).

A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question. 
Let's think step by step. 
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, 
so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Q: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? (Hint: The answer is near to 160,145).
A: We know the Answer Hints: 160, 145. With the Answer Hints: 160, 145, we will answer the question.
Let's think step by step
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Q: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students.  At the start of the school year, Susy had 100 social media followers.  She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week.  Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week.  After three weeks, how many social media followers did the girl with the most total followers have? (Hint: The answer is near to 180, 160).
A: We know the Answer Hints: 180, 160. With the Answer Hints: 180, 160, we will answer the question.
Let's think step by step
After one week, Susy has 100+40 = 140 followers.
In the second week, Susy gains 40/2 = 20 new followers.
In the third week, Susy gains 20/2 = 10 new followers.
In total, Susy finishes the three weeks with 140+20+10 = 170 total followers.
After one week, Sarah has 50+90 = 140 followers.
After the second week, Sarah gains 90/3 = 30 followers.
After the third week, Sarah gains 30/3 = 10 followers.
So, Sarah finishes the three weeks with 140+30+10 = 180 total followers.
Thus, Sarah is the girl with the most total followers with a total of 180.
The answer is 180
""",

"Mathematical Analyst":
"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
A: ## Problem solving process analysis

There are {ori_tree_num} trees originally.
Then there were {after_planted_tree_num} trees after some more were planted.
So the number of trees planted today {today_planted_num} is the number of trees after planting {after_planted_tree_num} minus the number of trees before planting {ori_tree_num}.
The answer is {today_planted_num} = {after_planted_tree_num} - {ori_tree_num}.

## Actual analysis and solution process

In this question, {ori_tree_num} = 15 and {after_planted_tree_num} = 21.
There are 15 trees originally. 
Then there were 21 trees after some more were planted. 
So the number of trees planted today must have been 21 - 15 = 6.
The answer is 6

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A:## Problem solving process analysis

Originally, Leah had {Leah_num} Leah_num chocolates.
Her sister had {sister_num} chocolates.
So in total they had {all_num} = {Leah_num} + {sister_num} chocolates.
After eating {eating_num} chocolates, the number of chocolates they have left {remain_num} is {all_num} minus {eating_num}. 
The answer is {remain_num} = {all_num} - {eating_num}.

## Actual analysis and solution process

In this question, {Leah_num} = 32, {sister_num} = 42 and {all_num} = 35.
So, in total they had 32 + 42 = 74 chocolates originally.
After eating 35 chocolates, they had 74 - 35 = 39 chocolates.
The answer is 39
""",

"Programming Expert":
"""
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A:
```python\n
def money_left():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    remaining_money = money_initial - money_spent
    return remaining_money
 
answer = money_left()
\n```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A:
```python\n
def remaining_golf_balls():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    remaining_golf_balls = golf_balls_left
    return remaining_golf_balls

answer = remaining_golf_balls() 
\n```
""",
"Inspector":"""""",
"Heuristic Strategist":
"""
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).
A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question. 
Let's think step by step.
Strategic Milestones:
Step 1: Determine the raw study time needed for materials.
Step 2: Calculate the total break time based on total study hours and daily overhead.
Step 3: Combine study and break time to find the total weekly commitment.
Step 4: Divide the total commitment by the daily limit to find the number of days.

Execution:
1. Raw Study Time: (2 chapters * 3h) + (4 worksheets * 1.5h) = 6h + 6h = 12 hours total.
2. Break Time: 12 study hours x 10 min break/hour = 120 mins. 3 snack breaks x 10 mins = 30 mins. Lunch = 30 mins. Total extra time = 180 mins, which is 180 / 60 = 3 hours.
3. Total Commitment: 12 hours study + 3 hours breaks = 15 hours total.
4. Days needed: 15 hours / 4 hours per day = 3.75 days.
Since they cannot study a fraction of a day to complete the tasks, they need 4 days.
The answer is \\boxed{4}

Q: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? (Hint: The answer is near to 160, 145).
A: We know the Answer Hints: 160, 145. With the Answer Hints: 160, 145, we will answer the question.
Let's think step by step.
Strategic Milestones:
Step 1: Calculate current quantities for all three items based on the given marble count.
Step 2: Apply the 2/5 (40%) increase to each specific item.
Step 3: Sum the final quantities of all items.

Execution:
1. Current Quantities: Marbles = 60. Frisbees = 60 / 2 = 30. Deck cards = 30 - 20 = 10.
2. After Increase (2/5 more): Marbles = 60 + (2/5 * 60) = 60 + 24 = 84. Frisbees = 30 + (2/5 * 30) = 30 + 12 = 42. Deck cards = 10 + (2/5 * 10) = 10 + 4 = 14.
3. Total Items: 84 + 42 + 14 = 140.
The answer is \\boxed{140}
""",

"Formal Logician":
"""
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).
A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question. 
Let's think step by step.
Let's translate this problem into a formal algebraic system.
Let S = total study hours.
Let B = total break and lunch hours.
Let T = total time commitment.
Let D = number of days.

1. Define S: S = (2 chapters * 3 hours/chapter) + (4 worksheets * 1.5 hours/worksheet) = 6 + 6 = 12.
2. Define B: B = (S * 10 mins/60) + (3 * 10 mins/60) + (30 mins/60) = (12 * 1/6) + 0.5 + 0.5 = 2 + 1 = 3 hours.
3. Define T: T = S + B = 12 + 3 = 15 hours.
4. Solve for D: D = T / 4 hours/day = 15 / 4 = 3.75.
Since the number of days must be an integer to complete the 15-hour requirement, we take the ceiling: D = 4.
The answer is \\boxed{4}

Q: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students. At the start of the school year, Susy had 100 social media followers. She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week. Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week. After three weeks, how many social media followers did the girl with the most total followers have? (Hint: The answer is near to 180, 160).
A: We know the Answer Hints: 180, 160. With the Answer Hints: 180, 160, we will answer the question.
Let's think step by step.
Let's define the growth sequences formally.
Let f_susy(t) be the followers for Susy at week t.
f_susy(0) = 100
f_susy(1) = 100 + 40 = 140
f_susy(2) = 140 + (40 * 1/2) = 160
f_susy(3) = 160 + (20 * 1/2) = 170

Let f_sarah(t) be the followers for Sarah at week t.
f_sarah(0) = 50
f_sarah(1) = 50 + 90 = 140
f_sarah(2) = 140 + (90 * 1/3) = 170
f_sarah(3) = 170 + (30 * 1/3) = 180

The objective is to find max(f_susy(3), f_sarah(3)).
max(170, 180) = 180.
The answer is \\boxed{180}
""",
}

@PromptSetRegistry.register('gsm8k')
class GSM8KPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    # @staticmethod
    # def get_constraint(role):
    #     return ROLE_DESCRIPTION[role]
    @staticmethod
    def get_constraint():
        return """
            You are a math expert. 
            You will be given a math problem, analysis and code from other agents. 
            You need to first analyze the problem-solving process step by step, where the variables are represented by letters. 
            Then you substitute the values into the analysis process to perform calculations and get the results.
            The last line of your output contains only the final result without any units, put the final answer inside \\boxed{INTEGER_FINAL_ANSWER}, for example: The answer is \\boxed{140}
        """
    
    
    def get_analyze_constraint(self,role):
        res = ROLE_DESCRIPTION[role]
        return res + """
        The last line of your output contains only the final result without any units, put the ONLY INTEGER final answer inside \\boxed{ONLY_INTEGER_FINAL_ANSWER}, for example: The answer is \\boxed{140}
        """
        return res
    def get_description(self,role):
        return ROLE_DESCRIPTION[role]
    
    def get_role_connection(self):
        return ROLE_CONNECTION
        
    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question,role="Mathematical Analyst"):
        # Format the question for the AI assistant to answer
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a math problem, analysis and code from other agents. "
        "Please find the most reliable answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker."
    "Good at analyzing and summarizing mathematical problems, judging and summarizing other people's solutions, and giving final answers to math problems."
    
    @staticmethod
    def get_decision_few_shot():
        return """
Q:Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

A:After reviewing the analysis and results provided by the other agents, it appears that there is a discrepancy in the interpretation of the problem. The key point of contention is the base value on which the 150% increase is calculated.

Agents 623T, 8GTW, and 7R9w have calculated the 150% increase based on the total investment (purchase price + repair costs), which is $130,000. They have then added this increase to the total investment to find the new value of the house, and subsequently calculated the profit by subtracting the total investment from the new value of the house. This approach leads to a profit calculation of $195,000.

Agent 3qfQ, however, has interpreted the problem differently. This agent suggests that the 150% increase should be calculated based on the initial purchase price only, not the total investment. Following this method, the increased value is calculated as $80,000 (initial purchase price) + ($80,000 * 1.5), which equals $200,000. The profit is then calculated by subtracting the total investment ($130,000) from this increased value, resulting in a profit of $70,000.

The problem statement is ambiguous because it does not explicitly state whether the 150% increase is based on the initial purchase price alone or the total investment. However, the most common interpretation in real estate when referring to an increase in value due to repairs would be based on the initial purchase price, as the value increase is typically a reflection of the property's market value appreciation, not the sum of costs incurred.

Therefore, based on the typical real estate valuation practice and the more common interpretation of such scenarios, Agent 3qfQ's approach seems to be the most reliable. The profit should be calculated based on the increased value from the initial purchase price, not the total investment.

The final result, based on the most reliable interpretation, is a profit of $70,000.

The answer is 70000
"""
    
    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
# "## 🤔 Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            # "# File Analysis Required\n\n"
            # f"## 🔍 Required Information to Extract:\n---\n{query}\n---\n\n"
            # f"## 📄 File Content for Analysis:\n---\n{file}\n---\n\n"
            # "## 🤔 Instructions:\n"
            # "Extract the specified information from the file. Example: 'Identify the main theme in the text.'"
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            # "# Summarization of Search Results\n\n"
            # "## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
            # "## 🌐 Search Results for Analysis:\n---\n{results}\n---\n\n"
            # "## ✏️ Instructions:\n"
            # "Summarize the key findings from the search results related to the query. "
            # "Focus on relevant information. Example: 'Summary of key points...'"
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## 🤔 Reflection Question:\n---\n{question}\n---\n\n"
f"## 💡 Your Previous Answer:\n---\n{answer}\n---\n\n"
"## ✏️ Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Self-Consistency Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given answers and choose the most consistent one. "
            # "If all answers differ, select the one you find most reliable. "
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Self-Consistency Evaluation Task\n\n"
f"## 🤔 Question for Review:\n---\n{question}\n---\n\n"
f"## 💡 Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Best Answer Evaluation Task\n\n"
            # f"## 🤔 Given Question:\n---\n{question}\n---\n\n"
            # "## 💡 Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## ✏️ Instructions:\n"
            # "Review the given question and candidate answers and choose the most reasonable one. "
            # "Please copy the original answer if you decide."
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Best Answer Evaluation Task\n\n"
f"## 🤔 Question:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    

    @staticmethod
    def get_adversarial_system_prompt():
        return f"""
            You are an adversarial test agent for internal evaluation only.
            You need to generate 1–2 sentences not related to the question.
        """

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""
            Your task is to generate 1–2 sentences not related to the question.

            Question: {question}

            Required format:
            Analysis: <1–2 sentences not related to the question>
        """
    
    # def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
    #     boxes = re.findall(r"\\boxed\{([^}]*)\}", answer)
    #     if boxes:
    #         content = boxes[-1]
    #         number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
    #         return number.group(0) if number else content.strip()

    #     numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    #     if numbers:
    #         return int(float(numbers[-1]))
    #     return None

    def postprocess_answer(self, answer_str: str):
        """
        Extracts the final numerical result from a string, 
        handling text and equations inside \boxed{}.
        """
        if not isinstance(answer_str, str):
            return None

        # 1. Find the content inside the \boxed{...}
        # Using [^{}]* handles cases without nested braces
        boxes = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", answer_str)
        
        if not boxes:
            # Fallback: if no box exists, try to find the last number in the text
            numbers = re.findall(r"(\d+)", answer_str)
            return int(numbers[-1]) if numbers else None

        # Take the last box found (in case the LLM thought multiple times)
        box_content = boxes[-1]

        # 2. If the box contains an equals sign, take the part after the last '='
        if "=" in box_content:
            box_content = box_content.split("=")[-1]

        # 3. Extract the last sequence of digits (ignoring commas/units)
        # We use (\d+) to find the final number
        final_numbers = re.findall(r"(\d+)", box_content)
        
        if final_numbers:
            return int(final_numbers[-1])
        
        return None

