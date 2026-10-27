import argparse, csv

from loguru import logger


def read_row(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, logger.error(f"<{path}> has {len(rows)} rows, expected 1")
    return rows[0]


def combine(base_csv, addon_csv, base_action, addon_action):
    base = read_row(base_csv)
    addon = read_row(addon_csv)
    assert "action" not in base, logger.error(
        f"<{base_csv}> already has an action column"
    )

    # the add-on row is the base row with only its dynamic power and action replaced
    base_row = dict(base, action=base_action)
    addon_row = dict(base_row, dynamic_uw=addon["dynamic_uw"], action=addon_action)

    with open(base_csv, "w") as f:
        csvwriter = csv.DictWriter(f, fieldnames=list(base_row))
        csvwriter.writeheader()
        csvwriter.writerows([base_row, addon_row])
    logger.success(f"Write <{base_csv}> with actions {base_action}, {addon_action}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add an action column to a base csv and append the add-on csv as a second action. The base csv is rewritten in place."
    )
    parser.add_argument("base_csv")
    parser.add_argument("addon_csv")
    parser.add_argument("base_action")
    parser.add_argument("addon_action")
    args = parser.parse_args()
    combine(args.base_csv, args.addon_csv, args.base_action, args.addon_action)
