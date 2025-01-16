import tkinter as tk
from tkinter import messagebox
import pandas as pd
import ast
import math

########################
# 1. LOAD DISEASE DATA
########################

def load_disease_data(excel_file_path):
    """
    Reads an Excel file with columns:
       'Disease Name' | 'Symptoms'
    and returns a dictionary:
       {
         "DiseaseA": ["symptom1", "symptom2", ...],
         "DiseaseB": ["symptomX", "symptomY", ...],
         ...
       }

    'Symptoms' column is expected to have a string like "['headache','nausea']".
    """
    df = pd.read_excel(excel_file_path)

    disease_symptoms = {}
    for _, row in df.iterrows():
        disease_name = row['Disease Name']
        symptoms_str = row['Symptoms']
        symptoms_list = ast.literal_eval(symptoms_str)
        # optional: normalize to lowercase
        symptoms_list = [s.lower().strip() for s in symptoms_list]

        disease_symptoms[disease_name] = symptoms_list

    return disease_symptoms


########################
# 2. INFORMATION GAIN LOGIC
########################

def best_symptom_to_ask_by_information_gain(candidates, disease_symptoms, asked_symptoms):
    """
    Picks the symptom with the highest ID3-style information gain.
    Returns None if no further symptom can discriminate.
    """
    n = len(candidates)
    if n <= 1:
        return None

    base_entropy = 0.0 if n <= 1 else math.log2(n)

    symptom_counts = {}
    for disease in candidates:
        for sym in disease_symptoms[disease]:
            if sym not in asked_symptoms:
                symptom_counts[sym] = symptom_counts.get(sym, 0) + 1

    if not symptom_counts:
        return None

    best_symptom = None
    best_gain = -1.0

    for symptom, count_yes in symptom_counts.items():
        count_no = n - count_yes

        yes_part = (count_yes / n) * math.log2(count_yes) if count_yes > 0 else 0.0
        no_part = (count_no / n) * math.log2(count_no) if count_no > 0 else 0.0

        conditional_entropy = yes_part + no_part
        info_gain = base_entropy - conditional_entropy

        if info_gain > best_gain:
            best_gain = info_gain
            best_symptom = symptom

    return best_symptom


def filter_diseases(candidates, disease_symptoms, symptom, answer):
    """
    Filters the list of candidates based on user input:
      - yes => keep only diseases that have 'symptom'
      - no  => keep only diseases that do NOT have 'symptom'
      - maybe => no filtering (keep them all)
    """
    ans = answer.lower().strip()
    if ans.startswith('y'):
        return [d for d in candidates if symptom in disease_symptoms[d]]
    elif ans.startswith('n'):
        return [d for d in candidates if symptom not in disease_symptoms[d]]
    else:
        # maybe or any other input => no filtering
        return candidates


########################
# 3. SCORING & TOP-5
########################

def compute_score(disease, disease_symptoms, asked_answers):
    """
    +1 if user said "yes" and disease has symptom
    +1 if user said "no"  and disease does NOT have symptom
    0  if user said "maybe"
    """
    score = 0
    disease_sym_list = disease_symptoms[disease]
    for sym, ans in asked_answers.items():
        ans = ans.lower().strip()
        if ans.startswith('y'):  # yes
            if sym in disease_sym_list:
                score += 1
        elif ans.startswith('n'):  # no
            if sym not in disease_sym_list:
                score += 1
        # "maybe" => no effect
    return score


def final_scoring(candidates, disease_symptoms, asked_answers, top_k=5):
    """
    Ranks each candidate by how well it matches the user's yes/no answers.
    Returns a list of (disease, score), sorted desc by score.
    """
    if len(candidates) == 1:
        # single candidate => trivially top-1
        return [(candidates[0], compute_score(candidates[0], disease_symptoms, asked_answers))]

    scored = []
    for d in candidates:
        sc = compute_score(d, disease_symptoms, asked_answers)
        scored.append((d, sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


########################
# 4. TKINTER GUI CLASS
########################

class AkinatorGUI:
    def __init__(self, root, disease_symptoms):
        self.root = root
        self.root.title("Disease Akinator")

        self.disease_symptoms = disease_symptoms
        self.candidates = list(disease_symptoms.keys())
        self.asked_symptoms = set()
        self.asked_answers = {}  # symptom -> user input

        # Create main frame
        self.frame = tk.Frame(root, padx=20, pady=20)
        self.frame.pack()

        # Question label
        self.question_label = tk.Label(self.frame, text="Welcome! Press 'Next' to begin.")
        self.question_label.pack(pady=10)

        # Buttons for yes, no, maybe
        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack()

        self.yes_btn = tk.Button(self.btn_frame, text="Yes", width=8,
                                 command=lambda: self.answer("yes"))
        self.yes_btn.pack(side=tk.LEFT, padx=5)

        self.no_btn = tk.Button(self.btn_frame, text="No", width=8,
                                command=lambda: self.answer("no"))
        self.no_btn.pack(side=tk.LEFT, padx=5)

        self.maybe_btn = tk.Button(self.btn_frame, text="Maybe", width=8,
                                   command=lambda: self.answer("maybe"))
        self.maybe_btn.pack(side=tk.LEFT, padx=5)

        # "Next" or "Start" button
        self.next_btn = tk.Button(self.frame, text="Next Question",
                                  command=self.next_question)
        self.next_btn.pack(pady=10)

        self.current_symptom = None

    def next_question(self):
        """
        Picks the next symptom using information gain,
        or finalizes if no more questions can help.
        """
        # If no candidates remain
        if len(self.candidates) == 0:
            messagebox.showinfo("Result", "No matching diseases found.")
            self.root.quit()
            return

        # Get next best symptom
        sym = best_symptom_to_ask_by_information_gain(self.candidates,
                                                      self.disease_symptoms,
                                                      self.asked_symptoms)
        if sym is None:
            # No further question can differentiate => show top 5
            self.show_top5_and_exit()
            return

        self.current_symptom = sym
        self.asked_symptoms.add(sym)

        # Update label
        self.question_label.config(text=f"Do you have '{sym}'?")

    def answer(self, ans):
        """
        Stores the user's answer for the current symptom,
        filters the candidates, and updates asked_answers.
        """
        if not self.current_symptom:
            # If user clicked yes/no/maybe before pressing 'Next Question' first
            messagebox.showwarning("Warning", "Please click 'Next Question' to get a symptom.")
            return

        # Record answer
        self.asked_answers[self.current_symptom] = ans

        # Filter candidates
        self.candidates = filter_diseases(self.candidates,
                                          self.disease_symptoms,
                                          self.current_symptom,
                                          ans)

        # Optionally, you could automatically go to the next question:
        self.next_question()

    def show_top5_and_exit(self):
        """
        When no more questions can help, compute top 5 diseases, display them, and quit.
        """
        top5 = final_scoring(self.candidates, self.disease_symptoms, self.asked_answers)
        if len(top5) == 1:
            msg = f"Likely disease: {top5[0][0]} (score: {top5[0][1]})"
        else:
            msg_list = []
            for i, (d, sc) in enumerate(top5, start=1):
                msg_list.append(f"{i}. {d} (score: {sc})")
            msg = "Top likely diseases:\n" + "\n".join(msg_list)

        messagebox.showinfo("Diagnosis Result", msg)
        self.root.quit()


########################
# 5. MAIN FUNCTION
########################

def main():
    excel_file_path = "C:\\Users\\Nishant Gupta\\Desktop\\unique_diseases_data.xlsx"  # <-- change path as needed
    disease_symptoms = load_disease_data(excel_file_path)

    root = tk.Tk()
    app = AkinatorGUI(root, disease_symptoms)
    root.mainloop()


if __name__ == "__main__":
    main()