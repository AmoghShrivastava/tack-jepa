# Nebius compute — DO NOT PROVISION YET

**Hard gate (PRD §9, Phase 5):** no Nebius GPU instance may be provisioned, launched,
or billed before Phases 0–4 are complete AND the user has given an explicit go-ahead
in that conversation. Phase 4's optional few-minute single-GPU loop validation also
requires explicit user confirmation first.

## Discipline for when provisioning does start (Phase 5+)

- Every launch command written in this directory MUST be accompanied, immediately
  adjacent, by its stop/deallocate command.
- Confirm current instance SKUs and pricing live via the Nebius console/CLI at
  provisioning time — never trust names/prices written earlier (PRD §11).
- Data generation (Genesis sim) is CPU-bound: run it on a CPU-only instance, never
  on GPU-billed time (PRD §6.4).
- Keep a cost log in this file: date, instance type, duration, purpose, cost.

## Cost log

*(empty — nothing has been provisioned)*
