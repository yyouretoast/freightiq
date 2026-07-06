def format_carrier_document(c):
    return (
        f"Carrier Name: {c['carrier_name']}\n"
        f"DOT Number: {c['dot_number']}\n"
        f"MC Number: {c['mc_number']}\n"
        f"HQ State: {c['hq_state']}\n"
        f"Service Regions: {', '.join(c['service_regions'])}\n"
        f"Equipment: {', '.join(c['equipment_types'])}\n"
        f"Specializations: {', '.join(c['cargo_specializations'])}\n"
        f"Safety Rating: {c['safety_rating']}\n"
        f"Years Operating: {c['years_operating']} years\n"
        f"Contact: {c['contact_email']}\n"
        f"Notes: {c['notes']}"
    )
