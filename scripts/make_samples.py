"""Generate realistic sample documents (PDF, DOCX, XLSX) for retrieval testing.

Usage: .venv/Scripts/python scripts/make_samples.py   -> writes samples/
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "samples"

HANDBOOK = {
    "1. Working hours": (
        "Standard working hours at Acme Robotics are 40 hours per week. Core hours are 10:00 to 15:00, "
        "during which all employees are expected to be reachable. Outside core hours, employees may "
        "organise their time freely as long as team commitments are met. Overtime must be approved in "
        "advance by the line manager and is compensated with time off in lieu at a rate of 1:1."
    ),
    "2. Vacation and leave": (
        "Full-time employees receive 27 days of paid vacation per calendar year. Part-time employees receive "
        "a pro-rated amount. Unused vacation days can be carried over until March 31 of the following year, "
        "after which they expire. Vacation requests longer than five consecutive days must be submitted at "
        "least four weeks in advance through the HR portal. Parental leave follows the statutory rules; "
        "Acme additionally pays a top-up of 10% of the gross salary during the first three months."
    ),
    "3. Sick leave": (
        "If you are sick, inform your manager before 9:00 on the first day of absence. A doctor's certificate "
        "is required from the third day of illness onward. Sick days do not count against vacation. "
        "If you fall ill during your vacation, the sick days are credited back when you submit a certificate."
    ),
    "4. Remote work": (
        "Employees may work remotely up to three days per week after completing their probation period. "
        "Fridays are core office days for the Berlin team. Working from abroad is allowed for up to 20 working "
        "days per year and must be registered with HR at least two weeks before travel because of tax and "
        "social security rules. A monthly home office allowance of 40 EUR is paid with the salary."
    ),
    "5. Equipment and IT security": (
        "Every new hire receives a laptop and a headset within their first week. Lost or stolen equipment "
        "must be reported to IT within 24 hours via the service desk. Company laptops must use full disk "
        "encryption and the password manager provided by IT. Passwords must be at least 14 characters long. "
        "Private USB drives may not be connected to company devices."
    ),
    "6. Travel and expenses": (
        "Business travel must be booked through the travel portal. Train travel is preferred for distances "
        "under 600 km. The daily meal allowance for domestic trips is 28 EUR. Hotel costs are reimbursed up "
        "to 150 EUR per night in major cities and 110 EUR elsewhere. Expense reports must be submitted within "
        "30 days of the trip, together with all original receipts."
    ),
    "7. Learning and development": (
        "Each employee has an annual learning budget of 1,500 EUR and three paid learning days. The budget "
        "can be used for courses, conferences, certifications and books. Requests above 500 EUR need approval "
        "from the department head. Unused learning budget does not carry over into the next year."
    ),
    "8. Performance reviews and salary": (
        "Performance reviews take place twice a year, in April and October. Salary adjustments are decided "
        "once a year in the October cycle and take effect on January 1. Employees receive a 13th month salary "
        "paid with the November payroll. The annual bonus depends on company results and individual goals."
    ),
    "9. Code of conduct": (
        "Acme Robotics does not tolerate harassment or discrimination of any kind. Concerns can be raised with "
        "your manager, HR, or anonymously through the ethics hotline. Gifts from suppliers worth more than "
        "50 EUR must be declared to the compliance team. Conflicts of interest must be disclosed in writing."
    ),
    "10. Leaving the company": (
        "The notice period is three months to the end of a month for both sides after probation; during the "
        "six-month probation period it is two weeks. All equipment must be returned on the last working day. "
        "Employees receive a written reference letter within two weeks after leaving."
    ),
}

SPEC_SECTIONS = [
    ("Overview", "The R7 is an autonomous mobile robot for warehouse picking. It navigates with LiDAR and "
                 "stereo cameras and integrates with common warehouse management systems through a REST API."),
    ("Battery and charging", "The R7 uses a 48 V lithium iron phosphate battery with a capacity of 2.4 kWh. "
                             "A full charge takes 90 minutes on the docking station; opportunity charging to 80% "
                             "takes 45 minutes. Typical runtime is 10 hours in mixed operation."),
    ("Safety", "The robot is certified to ISO 3691-4. It stops within 0.3 m at full speed when an obstacle is "
               "detected. Emergency stop buttons are located on all four sides. The maximum speed in shared "
               "zones with people is limited to 1.0 m/s."),
    ("Maintenance", "Wheels and brushes should be inspected every 500 operating hours. The LiDAR window must "
                    "be cleaned weekly. Firmware updates are delivered over the air every quarter."),
    ("Warranty", "The R7 comes with a 24-month warranty covering parts and labour. The battery is covered for "
                 "3,000 charge cycles or 36 months, whichever comes first. Damage from misuse is excluded."),
]

SPEC_TABLE = [
    ("Parameter", "Value"),
    ("Payload", "600 kg"),
    ("Maximum speed", "2.0 m/s"),
    ("Dimensions (L x W x H)", "1100 x 750 x 320 mm"),
    ("Weight", "185 kg"),
    ("Navigation", "LiDAR + stereo cameras, no floor markers needed"),
    ("IP rating", "IP54"),
    ("Operating temperature", "0 to 40 °C"),
]


def make_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Acme Robotics GmbH - Employee Handbook 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "This handbook summarises the main policies for employees in Germany. "
                         "In case of conflict, the employment contract and statutory law take precedence.")
    for title, body in HANDBOOK.items():
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(path))


def make_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading("Product specification: Acme R7 warehouse robot", 0)
    for title, body in SPEC_SECTIONS:
        doc.add_heading(title, level=1)
        doc.add_paragraph(body)
    doc.add_heading("Technical data", level=1)
    table = doc.add_table(rows=0, cols=2)
    for k, v in SPEC_TABLE:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = k, v
    doc.save(str(path))


def make_xlsx(path: Path) -> None:
    sales = []
    data = {
        "North": [(410, 1.23), (455, 1.37), (502, 1.51), (560, 1.68)],
        "South": [(300, 0.90), (318, 0.95), (290, 0.87), (344, 1.03)],
        "West": [(520, 1.56), (610, 1.83), (640, 1.92), (700, 2.10)],
        "East": [(150, 0.45), (170, 0.51), (210, 0.63), (260, 0.78)],
    }
    for region, quarters in data.items():
        for q, (units, rev) in enumerate(quarters, start=1):
            sales.append({"Region": region, "Quarter": f"Q{q} 2025", "Units sold": units,
                          "Revenue (EUR million)": rev, "Top product": "R7" if region != "East" else "R5"})
    prices = pd.DataFrame([
        {"Product": "R5 compact robot", "List price (EUR)": 38000, "Lead time (weeks)": 6},
        {"Product": "R7 warehouse robot", "List price (EUR)": 54000, "Lead time (weeks)": 8},
        {"Product": "Docking station", "List price (EUR)": 4200, "Lead time (weeks)": 2},
        {"Product": "Fleet manager licence (per robot/year)", "List price (EUR)": 1800, "Lead time (weeks)": 0},
        {"Product": "Extended warranty (+24 months)", "List price (EUR)": 6500, "Lead time (weeks)": 0},
    ])
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(sales).to_excel(xw, sheet_name="Sales 2025", index=False)
        prices.to_excel(xw, sheet_name="Price list", index=False)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    make_pdf(OUT / "employee_handbook.pdf")
    make_docx(OUT / "r7_product_spec.docx")
    make_xlsx(OUT / "sales_and_prices.xlsx")
    for p in sorted(OUT.iterdir()):
        print(p.name, p.stat().st_size, "bytes")
