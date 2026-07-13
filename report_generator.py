import os
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


class ReportGenerator:

    @staticmethod
    def generate(video_path, tracked_objects, unique_ids):

        # --------------------------------------------------
        # Create Reports folder
        # --------------------------------------------------
        report_folder = "Reports"
        os.makedirs(report_folder, exist_ok=True)

        video_name = Path(video_path).stem

        pdf_path = os.path.join(
            report_folder,
            f"{video_name}.pdf"
        )

        pdf = SimpleDocTemplate(pdf_path)

        styles = getSampleStyleSheet()

        elements = []

        # --------------------------------------------------
        # Title
        # --------------------------------------------------
        elements.append(
            Paragraph(
                "Forensic Detection Report",
                styles["Title"]
            )
        )

        elements.append(Spacer(1, 15))

        # --------------------------------------------------
        # Video Information
        # --------------------------------------------------
        elements.append(
            Paragraph(
                f"<b>Video :</b> {video_name}",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                f"<b>Total Unique Objects :</b> {len(unique_ids)}",
                styles["Heading2"]
            )
        )

        elements.append(Spacer(1, 20))

        # --------------------------------------------------
        # Detection Table
        # --------------------------------------------------
        table_data = [[
            "Sl No",
            "Snapshot",
            "Object Type",
            "First Seen",
            "Last Seen",
            "Confidence"
        ]]

        serial = 1

        for track_id, info in tracked_objects.items():

            image_cell = "N/A"

            try:

                if os.path.exists(info["image"]):

                    image_cell = Image(
                        info["image"],
                        width=60,
                        height=60
                    )

            except Exception:
                image_cell = "N/A"

            table_data.append([
                str(serial),
                image_cell,
                info["class"].title(),
                info["first_seen"],
                info["last_seen"],
                f"{info['max_conf']:.2f}"
            ])

            serial += 1

        table = Table(
            table_data,
            colWidths=[
                40,
                70,
                90,
                120,
                120,
                60
            ],
            rowHeights=[30] + [70] * (len(table_data) - 1)
        )

        table.setStyle(TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

            ('FONTSIZE', (0, 0), (-1, -1), 8),

            ('GRID', (0, 0), (-1, -1), 1, colors.black),

            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            ('ROWBACKGROUNDS',
             (0, 1),
             (-1, -1),
             [colors.whitesmoke, colors.lightgrey])

        ]))

        elements.append(table)

        elements.append(Spacer(1, 20))

        # --------------------------------------------------
        # Object Statistics
        # --------------------------------------------------
        elements.append(
            Paragraph(
                "Object Statistics",
                styles["Heading1"]
            )
        )

        object_stats = {}

        for _, info in tracked_objects.items():

            obj = info["class"]

            object_stats[obj] = object_stats.get(obj, 0) + 1

        stats_data = [["Object Type", "Count"]]

        for obj, count in sorted(object_stats.items()):

            stats_data.append([
                obj.title(),
                str(count)
            ])

        stats_table = Table(
            stats_data,
            colWidths=[220, 80]
        )

        stats_table.setStyle(TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),

            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

            ('GRID', (0, 0), (-1, -1), 1, colors.black),

            ('ALIGN', (0, 0), (-1, -1), 'CENTER')

        ]))

        elements.append(stats_table)

        elements.append(Spacer(1, 20))

        # --------------------------------------------------
        # Grand Total
        # --------------------------------------------------
        elements.append(
            Paragraph(
                f"<b>Grand Total Objects Detected :</b> {len(unique_ids)}",
                styles["Heading2"]
            )
        )

        elements.append(Spacer(1, 10))

        # --------------------------------------------------
        # Report Time
        # --------------------------------------------------
        elements.append(
            Paragraph(
                f"Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"]
            )
        )

        # --------------------------------------------------
        # Build PDF
        # --------------------------------------------------
        pdf.build(elements)

        print("=" * 60)
        print("PDF Report Generated Successfully")
        print(f"Video : {video_name}")
        print(f"Report : {pdf_path}")
        print(f"Total Objects : {len(unique_ids)}")
        print("=" * 60)