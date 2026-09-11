"""Run the fictional Cedar manufacturing review without a model or credentials."""

from examples.procedural_studio.data import build_demo


def main():
    payload = build_demo()
    case = payload["cases"][0]
    original = case["runs"]["original"]
    repaired = case["runs"]["repaired"]
    scores = payload["evaluation"]
    names = {"check": "reconcile", "answer": "review", "abstain": "hold"}

    def route(outcome):
        return " -> ".join(names.get(action, action) for action in outcome["actions"])

    print("Cedar infusion-pump manufacturing")
    print(f"Lot: {case['lot']}")
    print(f"Configuration: {case['configuration']}; {case['instruction']}")
    print("Fictional records. Packet routing is advisory; a human reviews the lot.")
    print()
    for record in case["records"]:
        print(f"{record['record_id']}: {record['title']}")
        print(f"  {record['narrative']}")
    print()
    print(f"Original route: {route(original)}")
    print(f"Original outcome: {original['answer']}")
    print(f"Repaired route: {route(repaired)}")
    print(f"Repaired outcome: {repaired['answer']}")
    print("Evidence: " + ", ".join(repaired["citations"]))
    print()
    for key, label in (("original", "Original"), ("repaired", "Repaired")):
        result = scores[key]
        print(f"{label} procedure: {result['passed']}/{result['total']} scripted test cases passed")
    print(f"Repair accepted: {payload['accepted']['accepted']}")
    print(f"Later shortcut accepted: {payload['rejected']['accepted']}")
    print(f"Rejected proposals retained: {len(payload['negative_memories'])}")
    print(f"Historical route: {route(case['runs']['historical'])}")
    print("Open the visual: python -m examples.procedural_studio")


if __name__ == "__main__":
    main()
