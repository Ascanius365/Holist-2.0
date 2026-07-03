from simple_chalk import chalk
from collections import Counter


class Amygdala:
    def __init__(self):
        self.action_history = []
        self.feedback_history = []
        self.max_history = 10
        self.repeat_threshold = 3
        self.dominance_threshold = 3
        self.warning = None


    def analyze_situation(self, response, observation):
        """
        Assesses the bot's current intent.
        Returns a warning if something is wrong, otherwise none.
        """
        if self.warning == "You died. A monster killed you in the night.":
            print(chalk.yellow("You died. A monster killed you in the night."))

        else:
            # 1. Put action and feedback in history
            action = response["action"]
            item = response.get("item")
            current_task = (action, item)
            self.action_history.append(current_task)

            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)

            current_feedback = observation.get("Tool feedback", "")
            self.feedback_history.append(current_feedback)

            if len(self.feedback_history) > self.max_history:
                self.feedback_history.pop(0)

            # We're looking at the history WITHOUT the action that was just added (i.e., :-1)
            history_without_current = self.action_history[:-1]

            # Count how often this "new" action occurred in the recent past.
            past_occurrences = history_without_current[-9:].count(current_task)

            # If the task has already appeared 3 times in the last 10 steps,
            # it is NOT considered new enough to clear the warning.
            is_truly_new = past_occurrences < 3

            if is_truly_new and len(self.action_history) >= 2:
                self.warning = None

            else:
                # Dominance check for ALL actions in the history
                task_counts = Counter(self.action_history)
                obsessed_tasks = []

                # We are reviewing all tasks that appeared in the last 10 steps
                for task, count in task_counts.items():
                    if count >= self.dominance_threshold:
                        task_name = f"{task[0]} {task[1] if task[1] else ''}".strip()
                        obsessed_tasks.append(f"'{task_name}' ({count}x)")

                # If we find any possessed tasks, create a combined alert
                if obsessed_tasks:
                    tasks_str = " and ".join(obsessed_tasks)
                    self.warning = (f"OBSESSION ALERT: You are stuck in a pattern involving {tasks_str}. "
                                    f"Your behavior is too repetitive. You MUST break this cycle and "
                                    f"switch to a completely different activity (e.g. exploring or crafting)!")

                # If the bot repeatedly receives the same (negative) feedback
                feedback = observation.get("Tool feedback", "")
                if feedback:
                    self.feedback_history.append(feedback)
                    if len(self.feedback_history) > self.max_history:
                        self.feedback_history.pop(0)

                    if self.feedback_history.count(feedback) >= self.repeat_threshold:
                        self.warning = f"REPETITIVE FEEDBACK: You keep getting '{feedback}'. Your current approach is not working!"

            # 3. Check for saturation
            if action == "craft" or action == "dig":
                parsed_inv = self.parse_inventory(observation.get("inventory", ""))
                if item in parsed_inv:
                    count = parsed_inv[item]
                    if count >= 10:
                        self.warning = (f"You already have {count}x {item}. "
                                        f"Stop {action}ing it and do something else!")

            # 4. Check for death
            event_history = observation.get("Event history", [])
            has_died = "death" in event_history

            if has_died:
                #print(chalk.yellow("You died. A monster killed you in the night."))
                self.warning = "You died. A monster killed you in the night."


    def parse_inventory(self, inventory_data):
        parsed_inv = {}

        # Falls es noch der rohe String aus observation_data["inventory"] ist
        if isinstance(inventory_data, str):
            # Wir suchen nach dem Teil in den Klammern []
            import re
            matches = re.findall(r"(\d+)x\s+([a-zA-Z0-9_]+)", inventory_data)
            for count, name in matches:
                parsed_inv[name] = int(count)

            return parsed_inv


    def inject_to_prompt(self):
        """Adds the amygdala warning to the prompt for the LLM."""
        if not self.warning:
            return None

        amygdala_block = f"\n\n### AMYGDALA SYSTEM WARNING\n{self.warning}\nYou MUST acknowledge this warning and change your behavior immediately."
        #print(chalk.yellow(amygdala_block))
        self.warning = None

        return amygdala_block