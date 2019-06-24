create schema if not exists semantic;
drop table if exists semantic.entities cascade;
drop table if exists semantic.events cascade;

create table semantic.entities as (
    with entities as (
    select
        distinct entity_id::varchar,
        latitude,
        longitude,
        city,
        state,
        metro,
        district,
        county,
        primary_subject,
        poverty::varchar,
        grade,
        teacher_prefix,
        (min(date) over (partition by entity_id))::timestamp as start_time,
        now()::timestamp as end_time

    from cleaned.projects
    )

    select * from entities
);

create table semantic.events as (
        with events as (
            select
                event_id::varchar,
                entity_id::varchar,
                type,
                price,
                school_charter,
                school_magnet,
                reach,
                eligible_double_your_impact_match,
                date,
                result


            from cleaned.projects
            order by
            date asc
    )
    select et.*, ev.type, ev.price, ev.reach, ev.date, ev.result, ev.eligible_double_your_impact_match,ev.school_charter,ev.school_magnet
    from semantic.entities et
    inner join events ev on et.entity_id = ev.entity_id
);
