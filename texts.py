# texts.py

TECHNICAL_DATA = [
    {
        "text": "Insert the M4 bolt into the servo motor mount. Tighten the fastener to 5 Nm.",
        "gt_entities": [
            {"name": "M4 bolt", "type": "Part"},
            {"name": "servo motor mount", "type": "Component"},
            {"name": "fastener", "type": "Part"},
            {"name": "5 Nm", "type": "Quantity"}
        ]
    },
    {
        "text": "The main control unit (MCU) must be reset after replacing the proximity sensor and running diagnostics.",
        "gt_entities": [
            {"name": "main control unit (MCU)", "type": "Machine"},
            {"name": "MCU", "type": "Machine"},
            {"name": "proximity sensor", "type": "Component"},
            {"name": "diagnostics", "type": "Function"}
        ]
    },
    {
        "text": "Run the system_diag() function to check the hydraulic pump pressure.",
        "gt_entities": [
            {"name": "system_diag()", "type": "Function"},
            {"name": "hydraulic pump", "type": "Machine"}
        ]
    },
    {
        "text": "Configure the robotic arm speed using parameter P50.",
        "gt_entities": [
            {"name": "robotic arm", "type": "Machine"},
            {"name": "P50", "type": "Parameter"}
        ]
    },
    {
        "text": "Replace the high-efficiency filter cartridge (model HEF-200) every 1000 hours of operation.",
        "gt_entities": [
            {"name": "high-efficiency filter cartridge", "type": "Part"},
            {"name": "HEF-200", "type": "Model_No"},
            {"name": "1000 hours", "type": "Time"}
        ]
    },
]