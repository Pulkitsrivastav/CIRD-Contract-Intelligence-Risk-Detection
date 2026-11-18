"""
generate_samples.py
Generates two sample NDA PDF documents in sample_documents/ for testing and viva demos:
1. sample_documents/standard_mutual_nda.pdf (Balanced mutual NDA covering all 10 clauses A-J)
2. sample_documents/high_risk_flawed_nda.pdf (Flawed NDA missing key protections & containing perpetual terms)
"""

import os
import fitz  # PyMuPDF


def create_pdf_from_sections(output_path: str, title: str, subtitle: str, sections: list[tuple[str, str]]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter
    margin_x = 54
    y = 60

    # Title block
    page.insert_text((margin_x, y), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.36))
    y += 22
    page.insert_text((margin_x, y), subtitle, fontsize=10, fontname="helv", color=(0.35, 0.4, 0.48))
    y += 28

    for heading, body in sections:
        if y > 690:
            page = doc.new_page(width=612, height=792)
            y = 60

        page.insert_text((margin_x, y), heading, fontsize=11, fontname="helv", color=(0.08, 0.12, 0.2))
        y += 16

        rect = fitz.Rect(margin_x, y, 612 - margin_x, 740)
        rc = page.insert_textbox(rect, body, fontsize=10, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
        # Estimate vertical space consumed based on character count
        lines_est = max(2, (len(body) // 82) + 2)
        y += lines_est * 14 + 10

    doc.save(output_path)
    doc.close()


STANDARD_NDA_SECTIONS = [
    ("1. Parties and Effective Date",
     "This Mutual Non-Disclosure Agreement (the 'Agreement') is entered into as of October 15, 2026, by and between "
     "Apex Cloud Technologies Pvt. Ltd. ('First Party') and Meridian Analytics Solutions ('Second Party'). "
     "Each party may act as a Disclosing Party or a Receiving Party with respect to Confidential Information."),
    ("2. Definition of Confidential Information",
     "Confidential Information means any non-public business, financial, technical, operational, or customer information, "
     "including trade secrets, software architecture, product roadmaps, and pricing models, disclosed by one party to the "
     "other in written, electronic, or oral form that is designated as confidential or reasonably understood to be confidential."),
    ("3. Permitted Use and Purpose",
     "The Receiving Party shall use the Confidential Information solely for the purpose of evaluating and pursuing a "
     "potential joint software integration partnership between the parties (the 'Permitted Purpose'). The Receiving Party "
     "shall not reverse engineer, disassemble, or commercially exploit Confidential Information for any unrelated purpose."),
    ("4. Exclusions from Confidentiality",
     "Confidential Information does not include information that: (a) is or becomes publicly available without breach of "
     "this Agreement by the Receiving Party; (b) was already known to the Receiving Party prior to disclosure without "
     "confidentiality restriction; (c) is independently developed by the Receiving Party without reference to the Disclosing "
     "Party's Confidential Information; or (d) is lawfully obtained from a third party authorized to disclose it."),
    ("5. Confidentiality Period",
     "The obligations of confidentiality and non-use under this Agreement shall remain in effect for a Confidentiality "
     "Period of three (3) years from the date of disclosure of the respective Confidential Information, and shall survive "
     "any earlier termination of this Agreement."),
    ("6. Return or Destruction of Information",
     "Upon termination of this Agreement or upon the written request of the Disclosing Party, the Receiving Party shall "
     "promptly return or securely destroy all tangible documents, electronic files, and copies containing Confidential "
     "Information, and provide written certification of destruction within fifteen (15) days."),
    ("7. Permitted Disclosure and Legal Compulsion",
     "The Receiving Party may disclose Confidential Information to its employees, officers, and legal or financial advisors "
     "who have a strict need to know for the Permitted Purpose and are bound by written confidentiality obligations. If "
     "compelled by law, subpoena, or court order to disclose Confidential Information, the Receiving Party shall provide "
     "prompt prior written notice to the Disclosing Party (where legally permitted) to seek a protective order."),
    ("8. Security and Safeguarding",
     "The Receiving Party shall protect Confidential Information against unauthorized access, theft, or unauthorized "
     "disclosure using at least the same degree of care it uses to protect its own confidential information of similar "
     "importance, and in no event less than reasonable care and industry-standard technical safeguards."),
    ("9. Liability and Remedies",
     "Both parties acknowledge that unauthorized disclosure of Confidential Information may cause irreparable harm for "
     "which monetary damages alone may be an inadequate remedy. Accordingly, either party shall be entitled to seek "
     "injunctive or equitable relief in addition to any other remedies available at law."),
    ("10. Governing Law and Jurisdiction",
     "This Agreement shall be governed by and construed in accordance with the laws of India, and the courts located in "
     "New Delhi shall have exclusive jurisdiction over any dispute arising out of or relating to this Agreement."),
    ("11. Termination",
     "Either party may terminate this Agreement at any time upon thirty (30) days' prior written notice to the other party. "
     "Termination shall not relieve either party of its confidentiality obligations with respect to Confidential Information "
     "disclosed prior to the effective date of termination, which shall survive termination as set forth in Section 5."),
]

FLAWED_NDA_SECTIONS = [
    ("1. Parties",
     "This Non-Disclosure Agreement is made between NovaCore Enterprise ('Disclosing Party') and Horizon Student Ventures "
     "('Receiving Party') for preliminary business discussions."),
    ("2. Definition of Confidential Information",
     "Confidential Information includes all information whatsoever shared by the Disclosing Party in written or electronic form, "
     "determined at the sole discretion of the Disclosing Party."),
    ("3. Permitted Use",
     "The Receiving Party may review the materials for evaluation of a potential vendor relationship."),
    ("4. Confidentiality Duration",
     "The Receiving Party shall keep all Confidential Information strictly secret in perpetuity and indefinitely forever, "
     "without any expiration date even if the information becomes widely known."),
    ("5. Remedies and Liability",
     "In the event of any suspected breach, the Receiving Party shall bear unlimited liability and pay automatic liquidated "
     "damages of USD 500,000, and waives all defenses against injunctive relief sought by the Disclosing Party."),
    ("6. Governing Law",
     "This Agreement is governed by the laws of the State of Delaware, and the Disclosing Party may sue in any court worldwide."),
]


if __name__ == "__main__":
    create_pdf_from_sections(
        os.path.join("sample_documents", "standard_mutual_nda.pdf"),
        "MUTUAL NON-DISCLOSURE AGREEMENT (STANDARD BASELINE SAMPLE)",
        "Sample Document 1: Balanced Mutual NDA covering Clauses A through J",
        STANDARD_NDA_SECTIONS,
    )
    create_pdf_from_sections(
        os.path.join("sample_documents", "high_risk_flawed_nda.pdf"),
        "NON-DISCLOSURE AGREEMENT (FLAWED / HIGH-RISK TEST SAMPLE)",
        "Sample Document 2: Intentionally omits key clauses and includes perpetual/one-sided terms",
        FLAWED_NDA_SECTIONS,
    )
    print("Generated sample PDFs in sample_documents/ successfully.")
