-- DESIGN ONLY. The offline runner does not execute this DDL or contact a business system.
-- Production: one first eligible application per persistent borrower; pilot is separate.
CREATE TABLE proposed_experiment_events (
    experiment_id VARCHAR NOT NULL,
    borrower_id VARCHAR NOT NULL,
    application_id VARCHAR NOT NULL,
    enrolled_at TIMESTAMP NOT NULL,
    experiment_phase VARCHAR NOT NULL CHECK (experiment_phase IN ('pilot','main')),
    eligibility_rule_version VARCHAR NOT NULL,
    common_business_eligibility BOOLEAN NOT NULL,
    assignment_arm VARCHAR NOT NULL CHECK (assignment_arm IN ('A','B')),
    assignment_probability_b DOUBLE NOT NULL CHECK (assignment_probability_b>0 AND assignment_probability_b<1),
    score_a DOUBLE NOT NULL,
    score_b DOUBLE NOT NULL,
    policy_a_version VARCHAR NOT NULL,
    policy_b_version VARCHAR NOT NULL,
    policy_a_threshold DOUBLE NOT NULL,
    policy_b_threshold DOUBLE NOT NULL,
    risk_score_stratum VARCHAR NOT NULL,
    approved BOOLEAN NOT NULL,
    decision_execution_matches_assignment BOOLEAN NOT NULL,
    principal DOUBLE,
    interest_collected DOUBLE,
    charged_off_principal DOUBLE,
    recoveries DOUBLE,
    funding_cost DOUBLE,
    servicing_cost DOUBLE,
    application_cost DOUBLE NOT NULL,
    mature_net_profit DOUBLE, -- interest - chargeoff + recovery - funding - servicing - application cost
    default_event BOOLEAN,
    dpd30_event BOOLEAN,
    dpd60_event BOOLEAN,
    dpd90_event BOOLEAN,
    outcome_due_at TIMESTAMP NOT NULL,
    outcome_complete BOOLEAN NOT NULL,
    finalized_at TIMESTAMP,
    PRIMARY KEY (experiment_id,borrower_id),
    UNIQUE (experiment_id,application_id)
);
-- Returned principal is never revenue. NULL mature outcomes are never coded as zero.
-- Rejected applications keep their arm and cost. Cashflow definitions and recovery
-- horizon must be fixed before enrollment; financial ledger is required in a real trial.
