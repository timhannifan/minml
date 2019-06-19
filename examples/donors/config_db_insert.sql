INSERT INTO raw_projects (projectid,teacher_acctid,school_city,school_state,school_county,school_charter,primary_focus_subject,resource_type,poverty_level,grade_level,total_price_including_optional_support,students_reached,eligible_double_your_impact_match,date_posted,datefullyfunded)
SELECT DISTINCT projectid,teacher_acctid,school_city,school_state,school_county,school_charter,primary_focus_subject,resource_type,poverty_level,grade_level,total_price_including_optional_support,students_reached,eligible_double_your_impact_match,date_posted,datefullyfunded
FROM raw;
