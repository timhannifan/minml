create schema if not exists cleaned;
drop table if exists cleaned.projects cascade;

create table cleaned.projects as (
    with cleaned as (
        select
            projectid::varchar as event_id,
            teacher_acctid::varchar as entity_id,
            btrim(replace(regexp_replace(btrim(lower(school_city)), '\s{2,}|,|\.',''), $$'$$,'')) as city,
                btrim(lower(school_state)) as state,
                btrim(replace(regexp_replace(btrim(lower(school_county)), '\s{2,}|,|\.',''), $$'$$,'')) as county,
                case when
                    primary_focus_subject is null then 'unknown'
                    else btrim(lower(primary_focus_subject))
                end as primary_subject,
                case when
                    resource_type is null then 'other'
                    else btrim(lower(resource_type))
                end as type,
                btrim(replace(regexp_replace(btrim(lower(poverty_level)), 'poverty',''), $$'$$,'')) as poverty,
                case
                    when grade_level = 'Grades 3-5' then 'primary'
                    when grade_level = 'Grades PreK-2' then 'pre'
                    when grade_level = 'Grades 6-8' then 'middle'
                    when grade_level = 'Grades 9-12' then 'high'
                    else 'other'
                end as grade,
                total_price_including_optional_support::real as price,
                students_reached::integer as reach,
                date_posted::timestamp as date,

                case
                    when school_magnet = 't' then 1
                    when school_magnet = 'f' then 0
                    else -1
                end as school_magnet,
                case
                    when school_charter = 't' then 1
                    when school_charter = 'f' then 0
                    else -1
                end as school_charter,
                case
                    when eligible_double_your_impact_match = 't' then 1
                    when eligible_double_your_impact_match = 'f' then 0
                    else -1
                end as eligible_double_your_impact_match,

                case
                    when DATE_PART('day', datefullyfunded - date_posted) > 60 then 1
                    else 0
                end as result

            from raw_projects
        )
    select * from cleaned
);
