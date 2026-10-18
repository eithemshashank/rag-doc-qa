"""Creates the two fictional PDFs used by the evaluation.

The institute and the company are made up, so every answer is known exactly
and nothing is copied from a real organisation. Run:  python eval/make_docs.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent / "docs"

HANDBOOK = ("greenfield_student_handbook.pdf", "Greenfield Institute of Technology - Student Handbook", [
    ("1. Attendance Rules", [
        "Students must have at least 75% attendance in every subject to be allowed to sit the end-semester examinations.",
        "Students with attendance between 65% and 74% may apply for condonation. The condonation fee is Rs. 500 per subject.",
        "Students with attendance below 65% are detained and must repeat the subject in the next academic year.",
        "Absence due to illness must be supported by a medical certificate, submitted within 7 days of returning to class.",
    ]),
    ("2. Library", [
        "Every student may borrow up to 4 books at a time for a period of 14 days.",
        "A book returned late is charged a fine of Rs. 2 per day for each book.",
        "The reading room is open from 8:00 AM to 8:00 PM, Monday to Saturday, and is closed on Sundays.",
        "A lost book must be replaced, or the student pays twice the price of the book.",
    ]),
    ("3. Examinations and Results", [
        "Each subject has two internal tests per semester. Each internal test carries 30 marks.",
        "The end-semester examination carries 70 marks. To pass a subject a student needs at least 40% of the total marks.",
        "A student who wants a revaluation must apply within 10 days of the result date. The revaluation fee is Rs. 750 per subject.",
        "Students must carry their identity card to every examination hall.",
    ]),
    ("4. Hostel and Fees", [
        "Hostel residents must be inside the hostel by 9:30 PM every night.",
        "The mess fee is Rs. 4,200 per month and must be paid before the 5th day of the month.",
        "Visitors are allowed only on Sundays between 4:00 PM and 6:00 PM.",
        "If a student withdraws within 15 days of the start of the semester, 80% of the tuition fee is refunded.",
    ]),
])

POLICY = ("nimbus_leave_and_remote_policy.pdf", "Nimbus Cloud Services - Leave and Remote Work Policy", [
    ("1. Leave Entitlement", [
        "Every full-time employee receives 24 days of paid annual leave per calendar year.",
        "In addition, employees get 8 days of casual leave and 12 days of sick leave each year.",
        "Up to 10 days of unused annual leave can be carried forward to the next year. Any extra days lapse.",
        "Leave of more than 3 consecutive days must be requested from the manager at least 5 working days in advance.",
    ]),
    ("2. Hybrid and Remote Work", [
        "Nimbus follows a hybrid model. Employees work from the office on Tuesday, Wednesday and Thursday.",
        "On remote days the core working hours are 11:00 AM to 4:00 PM, when everyone must be reachable.",
        "A remote work internet stipend of Rs. 1,500 per month is paid with the salary.",
        "Working remotely from a different city is allowed for a maximum of 20 days per year, with manager approval.",
    ]),
    ("3. Travel and Expenses", [
        "Business flights must be booked in economy class.",
        "The daily meal allowance for domestic travel is Rs. 800.",
        "Expense claims must be submitted with receipts within 30 days of the expense.",
        "Any single expense above Rs. 10,000 needs written approval from a director.",
    ]),
    ("4. Probation and Exit", [
        "New employees are on probation for the first 6 months.",
        "The notice period is 60 days for confirmed employees and 30 days during probation.",
        "The company laptop and ID card must be returned on the last working day.",
        "Gratuity is paid to employees who have completed at least 5 years of service.",
    ]),
])


def build(filename, title, sections):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4, title=title)
    story = []
    for i, (heading, lines) in enumerate(sections):
        if i:
            story.append(PageBreak())  # one section per page, so page numbers are known
        story.append(Paragraph(title, styles["Title"]))
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 8))
        for line in lines:
            story.append(Paragraph(line, styles["BodyText"]))
            story.append(Spacer(1, 8))
    doc.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in (HANDBOOK, POLICY):
        build(*spec)
        print("wrote", OUT / spec[0])
