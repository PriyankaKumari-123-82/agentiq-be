# Organization & Collaboration — Design Doc

Design for multi-organization membership, email invites, and admin-defined roles. It extends the
existing `Organization`/`Role`/`UserRole` models introduced in
`ai_agent/alembic/versions/2026_06_24_0000_add_organizations.py` and
`2026_06_29_0000_add_roles_tables.py`, which originally assumed **one organization per user**.

## 0. Implementation status

Built incrementally, one reviewable feature at a time.

- **Feature 1 — Foundation (shipped)**: data model (§4), permission layer (§5), and the
  existing `register`/`organization`/`roles` endpoints updated to the new membership model,
  plus `GET /organizations`, `GET /organizations/{org_id}`, `POST /organizations/{org_id}/switch`,
  and `GET /permissions`. Frontend: org switcher (in the `Layout.tsx` account dropdown and on the
  Settings page), "create organization" wired up, role badge fixed.
- **Feature 2 — Members management (shipped)**: `GET/PATCH/DELETE /organizations/{org_id}/members/...`.
  Frontend: a "Members" tab in Settings (list, change role, remove, leave).
- **Feature 3 — Invitations (shipped)**: `POST/GET /organizations/{org_id}/invitations`,
  `DELETE .../invitations/{invitation_id}`, `POST .../invitations/{invitation_id}/resend`, plus
  public `GET /invitations/{token}` and `POST /invitations/{token}/accept`. `identity.py` gained
  `email_verified` on `create_identity_user`. Frontend: invite form + invitations list (with
  resend/cancel) added to the Members tab, and a new public `/invitations/accept` page.

None of these passes have been run against a live DB — migration + import/syntax-level checks
only (see "Verification" below); each feature is followed by a review checkpoint before the next
starts.

### Deviations / additions found during Feature 1 implementation

- **`get_org_membership` dependency added to `core/deps.py`**, alongside the
  `get_active_membership`/`require_permission` pair originally sketched in §5. It resolves the
  caller's membership for an explicit `{org_id}` path parameter rather than their currently
  *active* org — used by `GET /organizations/{org_id}` and the switch endpoint, since viewing or
  switching into an org you belong to shouldn't require it to already be active.
- **Pre-existing frontend bug fixed**: `UserDetails.roles?: string[]` (frontend) never matched the
  backend's actual `role: str | None` (singular) field from `GET /user` — so the role badge on the
  Settings page was always empty. Found while wiring Feature 1's role plumbing end-to-end; renamed
  to `role` on both the type and `AuthContext.tsx`.
- **`roles.py`'s delete-role guard already checks pending invitations** (not just active members),
  per the edge case identified in §9 — implemented in Feature 1 even though invitation *creation*
  doesn't ship until Feature 3, so the check is dormant (always passes, no rows exist yet) but
  won't be forgotten later.

### Deviations / additions found during Feature 2 implementation

- **`GET /user` now also returns `permissions: list[str]`** — the caller's effective permission
  list in their active org (`["*"]` for owner/superuser, else the role's `permissions`). Not in
  the original §6 API table. Needed because the frontend has no other way to know "can this user
  manage members" without hardcoding role names (which the whole design deliberately avoids) —
  the Members tab uses it to decide whether to show role-change/remove controls at all, rather
  than showing them and relying on a 403 to fail silently.
- **`core/deps.py` gained `get_effective_permissions`, `user_has_permission`, and
  `require_org_permission`** — the `{org_id}`-scoped counterpart to `require_permission` that §5
  deferred to this feature. `remove_member` needed a permission check with a self-removal bypass
  (any member can leave, but only `manage_members` holders can remove *others*), which doesn't fit
  a single dependency, so `members.py` calls `user_has_permission` directly there while `PATCH
  .../members/{user_id}` (no such exception) uses `require_org_permission("manage_members")` as a
  plain dependency.
- **Changing the owner's role is blocked (403)**, not just removing them — §9 already said "can't
  be removed or demoted," this makes the demotion half explicit in `update_member_role`.

### Deviations / additions found during Feature 3 implementation

- **No separate `services/invitations.py`** — §12 originally suggested one, but by the time this
  feature landed, `organization.py`/`roles.py`/`members.py` had already established the
  codebase's actual convention: business logic lives directly in the router, and `services/` is
  reserved for wrapping external integrations (Keycloak in `identity.py`). Kept `invitations.py`
  consistent with its siblings instead of introducing the one router with a separate service
  layer.
- **New-user acceptance auto-logs the caller in.** `POST /invitations/{token}/accept` returns
  `access_token`/`refresh_token` (by internally calling `authenticate_user` right after
  `create_identity_user`) when the invitee didn't have an account yet — not in the original §6
  API table. Without this, someone would set a password to join and then immediately have to log
  in again with it, which is friction the accept flow can just remove for free since the backend
  already has the password in hand.
- **`resend` also revives EXPIRED invitations**, not just PENDING ones (only CANCELLED/ACCEPTED
  are rejected, 400). §7 described resend as refreshing token+expiry without pinning down which
  starting statuses are valid; allowing EXPIRED avoids forcing a cancel-then-recreate cycle for
  the common "sorry, forgot to click it in time" case.
- **`axiosInstance.ts`'s 401 interceptor now excludes `/invitations`** (alongside the existing
  `/login`/`/register` exclusions) — found while building the accept-invite page. A 401 from
  `POST /invitations/{token}/accept` (existing-email invite, not yet logged in) is an expected,
  recoverable state on a public page, not an expired-session event; without this fix the
  interceptor would have force-cleared tokens and hard-redirected, breaking the "log in, then
  come back" flow for users who *were* already logged in as a different account.
- **Accept-invite is deliberately not wrapped in `refreshUser()` + navigate for the new-user
  path** — it hard-reloads to `/dashboard` after writing tokens to `localStorage` directly
  (mirroring how `AuthContext.login()` stores tokens), rather than trying to thread the new
  session into React state in place. Simpler and more robust than replicating `AuthProvider`'s
  bootstrap logic a second time in a one-off page.

### Bugs found during manual end-to-end testing (post-implementation)

- **`TypeError: can't compare offset-naive and offset-aware datetimes`** in both
  `GET /invitations/{token}` and `POST /invitations/{token}/accept` — every datetime column in
  this codebase is a naive `TIMESTAMP` (see `models.py`'s `datetime.now(timezone.utc)`
  `default_factory` pattern, used everywhere, none declared `timezone=True`), so
  `invitation.expires_at` comes back from the DB naive even though it was written as a tz-aware
  UTC value. Comparing it directly against a fresh `datetime.now(timezone.utc)` crashed with a
  500 on *every* invitation link — this wasn't an environment issue, every invite was broken.
  Fixed with a `_now_naive()` helper used at both comparison sites (assignment-only `now()` calls
  elsewhere in the file didn't need it — storing a tz-aware value into a naive column is harmless,
  only comparing a round-tripped value against a fresh aware one breaks).
- **Frontend masked the real error**: `AcceptInvite.tsx`'s error state always displayed
  "This invitation link is invalid" whenever `loadError` was set, regardless of what the actual
  error was (a 500 got the same message as a 404) — `loadError` already held the correct
  translated string from the `.catch()` handler, but the JSX re-derived a hardcoded message
  instead of rendering it. This is *why* the 500 above took several rounds to diagnose — the UI
  reported a plausible-but-wrong cause ("invalid link") that pointed everyone toward environment
  config instead of the actual crash. Fixed to render `{loadError}` directly.
- **`sonner`'s `<Toaster />` was never mounted anywhere in the app** — `App.tsx` mounts the
  shadcn/Radix `Toaster` (`@/components/ui/toaster`) and a third custom `ToastContainer`, but
  never the `sonner`-backed one (`@/components/ui/sonner.tsx`), even though `Settings.tsx` (and
  now `Organization.tsx`/`OrgSwitcher.tsx`) call `toast.success()`/`toast.error()` from the
  `sonner` package throughout. Those calls never threw — they just had no renderer subscribed, so
  every one of them was a silent no-op. This is why "invite already pending" (a real 409 the
  backend already returned correctly) appeared to do nothing. Pre-existing bug, not introduced by
  this feature, but this feature's reliance on `sonner` toasts is what surfaced it. Fixed by
  mounting `<SonnerToaster />` in `App.tsx` alongside the other two.
- **Invite email showed the inviter's raw email** ("prashant@prismberry.com has invited you...")
  instead of a name — added a `_display_name()` helper (first + last name, falling back to email)
  used in both the email body and the accept-invite landing page's inviter line.
- **Invitations list showed one row per historical invite record**, not one per email — e.g.
  cancelling and re-inviting the same address left both the old (cancelled) and new rows visible.
  The API intentionally keeps full history (§9's audit-trail reasoning), but the UI only needs the
  latest status per email; `Organization.tsx` now dedupes client-side, keeping the first (most
  recent, since the API returns `created_at desc`) row per email.

### Organization rename (added post-launch)

Testing surfaced an organization named "User" — not a bug, but genuine leftover data: the
pre-multi-org migration (`2026_06_24_0000_add_organizations.py`) backfilled every existing user's
organization name from their `first_name` column, and there was no way to fix it afterward since
organizations could be created but never renamed. Added:
- `PATCH /organizations/{org_id}` in `organization.py`, gated by the `manage_organization`
  permission (already in the vocabulary since §5, never previously wired to an endpoint).
- A rename UI (pencil icon next to the org name) on the `/organization` page header, visible only
  to callers with `manage_organization`.

### "Zero organizations" was never a handled state (found via leave-org testing)

Manually testing "leave organization" surfaced a real gap: nothing in the app treated a user
belonging to *zero* organizations as valid, even though leaving your only org is a completely
normal way to reach that state.

- **`GET /user` used to 404 whenever `active_organization_id` was `NULL`** — its query
  inner-joined `Organization` on that column, so a `NULL` matched nothing and the whole endpoint
  came back "Organization not found." Every other piece of org context
  (`role`, `permissions`, `dbId`) was fetched in that same call, so this one 404 cascaded into
  the *entire* `AuthContext` bootstrap failing non-fatally — which silently left `user.dbId`
  undefined, which is why unrelated pages (Agents, Documents) started reporting "failed to
  fetch": they call those APIs keyed on `dbId`, which had never actually been populated. Fixed by
  making `organization_id`/`organization_name` on `UserInfoResponse` nullable and skipping the
  join entirely when there's no active org, instead of treating it as an error.
- **`OrgSwitcher.tsx` returned `null` when `organizations.length === 0`** — hiding the *only*
  place "Create Organization" lives, exactly when a user needs it most. Now renders a distinct
  "Create Organization" button instead of disappearing.
- **Leaving your last org showed a false failure.** `handleConfirmRemoveMember` called
  `removeMember()` (which succeeded) then `refreshUser()` (which threw, because of the `GET
  /user` bug above) — and since both were in the same `try`, the `catch` reported "could not
  leave organization" even though the leave had already gone through server-side, and the UI
  never updated to reflect it (matching "I still see the organization after leaving"). Restructured
  so a refresh failure after a successful leave is non-fatal — the success toast and state update
  now happen immediately after `removeMember()` resolves, with `refreshUser()` as a best-effort
  follow-up in its own try/catch.
- **The `/organization` page had no explicit empty state** for zero orgs — it would just render
  empty tabs with confusing "no members" messages. Added an early return with a clear "you don't
  belong to an organization yet" card pointing at the topbar switcher.

### Post-Feature-3 frontend restructuring

After Feature 3 shipped, org/member/role management was still tucked inside a Settings tab —
usable but not very discoverable, and there was no dedicated way to switch organizations short of
opening a dropdown menu. Reworked based on direct feedback into:

- **A dedicated `/organization` page** (`src/pages/Organization.tsx`), linked from a new
  "Organization" item in `Sidebar.tsx` — pulled the Members/Invitations UI out of Settings
  entirely, and added the role-management UI that never existed on the frontend before (the
  `/roles` API has supported full CRUD since Feature 1, but nothing called it) — create/edit/
  delete custom roles with permission checkboxes sourced from `GET /permissions`, matching the
  backend's permission-based model rather than hardcoding role names anywhere in the UI.
- **`OrgSwitcher.tsx`** — an always-visible switcher in the topbar next to `LanguageSelector`,
  replacing the switcher that had been folded into the account dropdown menu (functionally
  present since Feature 1, but easy to miss). Also carries the "create new organization" mini-form
  Slack/Notion-style, so creating and switching orgs live in one place instead of being split
  between a dropdown and a Settings form.
- **Settings.tsx reverted to its pre-Feature-1 shape** for the Organization card — read-only org
  name display plus a link to `/organization` — since switching/creating/managing now live
  elsewhere and duplicating them in Settings would just be two sources of the same UI.

### Verification performed

Docker/DB access wasn't available in this pass (verification deferred to the user running
`alembic upgrade head` locally). What was checked instead: `ast.parse` on every changed Python
file, a full-repo grep for stale references to removed symbols (`UserRole`,
`User.organization_id`, `Organization.members`, `Role.users`) to confirm nothing outside the
touched files breaks at import time, and for the frontend restructuring, a paren/brace-balance
check plus an unused-import sweep across every changed `.tsx`/`.ts` file (no compiler available
in this environment to do a real typecheck).

## 1. Requirements covered

- Signing up always makes you an Admin (of your own new organization).
- Admins invite users by email, assigning a role up front; the role can be changed later.
- Invited users who don't have an account yet must sign up, and are then linked to the inviting
  organization automatically.
- Users can create additional organizations of their own and become Admin of each one.
- Admins can see who they've invited and whether the invite was accepted.
- Admins can cancel a pending invite.
- Users who belong to multiple organizations can switch between them.

## 2. What already exists (don't rebuild)

| Piece | File | Status |
|---|---|---|
| `Organization` model | `ai_agent/models.py:35` | thin — id, name, timestamps |
| `Role` model (org-scoped, unique per org+name, `is_system` flag) | `ai_agent/models.py:115` | exists, CRUD already live |
| `UserRole` link table (user↔role, many-to-many) | `ai_agent/models.py:78` | exists |
| `User.organization_id` | `ai_agent/models.py:98` | **single FK — the core limitation** |
| `POST /register` | `api/v1/user.py:122` | creates org + `ADMIN` role + user in one transaction |
| `GET/POST /organization` | `api/v1/organization.py` | single-org only; `POST` 400s if user already has an org |
| `GET/POST/PATCH/DELETE /roles` | `api/v1/roles.py` | admin-gated CRUD, name-based admin check |
| SES email sending | `core/email_service.py` | `send_html_email()`, used today for password-change notices |
| Keycloak identity | `services/identity.py` | user creation, login, verify-email — all via Admin REST API |

Two bugs/gaps found while designing this that get fixed as part of the work:
- The bootstrapped `ADMIN` role is created with `is_system=False` (`user.py:171`) — an admin can
  currently rename or delete their own only admin role via the existing `roles.py` endpoints and
  lock themselves out.
- `User.organization_id` being a single column is why "switch organization" isn't possible today
  — it has to become a membership table.

## 3. Key design decisions

**a. Multi-org via a membership table, single "active org" pointer on `User`.**
A user can belong to many orgs but only one is "active" at a time (drives which org's data you
see). Rather than requiring the frontend to pass an `X-Organization-Id` header on every request
(more "correct" REST, but a much bigger frontend lift and a bigger diff across every existing
org-scoped endpoint), `active_organization_id` is persisted on `User` — same pattern already in
use today, just backed by a real membership table underneath. Switching orgs is one `PATCH` call.

**b. Custom roles + a fixed, small permission vocabulary.**
Role *names* stay fully admin-defined (already true today). But some actions always have to be
gated (who can invite, who can change roles) — that requires the app to define what a permission
*means*, even if it stays silent about what to call the role that has it. So:
- Add `Role.permissions: list[str]` (JSON column), e.g.
  `["manage_members", "manage_roles", "manage_organization"]`.
- The bootstrapped org-creator role is `is_system=True`, name `"Admin"`, `permissions=["*"]`,
  can't be renamed/deleted (fixes the bug above).
- Admins create additional custom-named roles and pick which permissions to grant, from that
  fixed vocabulary — e.g. a role called "Billing Contact" with only `["manage_organization"]`.
- Authorization dependencies check permissions (`require_permission("manage_members")`), never
  role names — replacing `roles.py`'s current `Role.name == "ADMIN"` string check.

**c. Consolidate `UserRole` into the new membership table.**
Today `UserRole` is a many-to-many user↔role link, but in practice each user has exactly one role
per org. Role assignment folds into the new `organization_memberships` table as a single
`role_id` column ("assign role before inviting, change it later" implies one role per membership,
not a set). This drops the `user_role` table — it's redundant once membership carries the role,
and keeping both would create two sources of truth for "what role does this user have."
`roles.py`'s delete-role guard (which currently checks `user_role`) moves to checking
`organization_memberships`.

**d. `is_owner` flag survives permission misconfiguration.**
The user who created the org (via signup or "create organization") is flagged `is_owner=True` on
their membership — independent of role permissions, meaning "can't be removed, can't be demoted
below full access by anyone else." Mirrors the existing `User.is_superuser` global escape hatch
already in the codebase (`roles.py:46`).

**e. Invite tokens are hashed at rest.**
Like a password-reset token, only the SHA-256 hash is stored (`token_hash`); the raw token goes
out only in the email link — a DB read alone can't leak a live invite link.

**f. Invited users skip separate email verification.**
Clicking a unique, emailed invite link already proves inbox ownership. For brand-new users
accepting an invite, `create_identity_user` gets called with `emailVerified=True` directly (a
small optional-param addition to `identity.py`) instead of triggering Keycloak's separate
`VERIFY_EMAIL` flow.

## 4. Data model changes

### New table: `organization_memberships`
| column | type | notes |
|---|---|---|
| id | PK | |
| user_id | FK users | |
| organization_id | FK organizations | |
| role_id | FK roles, NOT NULL | |
| is_owner | bool, default false | org creator only |
| is_active | bool, default true | soft-delete on removal (matches `Document.is_active` pattern) |
| removed_at | datetime, nullable | |
| joined_at | datetime | when accepted |
| created_at / updated_at | datetime | |

Unique constraint: `(user_id, organization_id)`.

### New table: `organization_invitations`
| column | type | notes |
|---|---|---|
| id | PK | |
| organization_id | FK organizations | |
| email | str, indexed, lowercased | |
| role_id | FK roles, NOT NULL | must be chosen before inviting |
| invited_by_user_id | FK users | |
| token_hash | str, unique, indexed | SHA-256 of the raw token |
| status | enum: PENDING / ACCEPTED / CANCELLED / EXPIRED | |
| expires_at | datetime | default 7 days |
| accepted_at | datetime, nullable | |
| accepted_user_id | FK users, nullable | |
| cancelled_at | datetime, nullable | |
| created_at / updated_at | datetime | |

Postgres partial unique index: `(organization_id, lower(email)) WHERE status = 'PENDING'` —
prevents duplicate open invites without blocking re-inviting after cancellation.

### Altered: `Role`
- add `permissions: list[str]` (JSON, default `[]`).
- backfill: existing `ADMIN` roles get `permissions=["*"]`, `is_system=True`.

### Altered: `User`
- rename `organization_id` → `active_organization_id` (same FK, nullable).
- drop `roles` relationship (via `UserRole`) and `organization` singular relationship.
- add `memberships: List["OrganizationMembership"]`.

### Dropped: `user_role` table (superseded by `organization_memberships.role_id`, see 3c).

### Migration
`ai_agent/alembic/versions/2026_07_03_0000_add_organization_collaboration.py`, following the
exact style of `2026_06_29_0000_add_roles_tables.py`:
1. Add `roles.permissions` JSON column; backfill `is_system=true, permissions='["*"]'` for
   existing `ADMIN` rows.
2. Create `organization_memberships`; backfill one row per existing user (`is_owner=true`,
   `role_id` = their current `ADMIN` role via `user_role` join, `joined_at = users.created_at`) —
   safe because today's data is exactly one org/one admin per user.
3. Create `organization_invitations` + the partial unique index.
4. Rename `users.organization_id` → `active_organization_id`.
5. Drop `user_role` table.

Each step is reversible in `downgrade()`, matching existing migration conventions.

## 5. Permission/authorization layer

New `ai_agent/core/deps.py` (extracted and generalized from `roles.py`'s private
`_get_caller`/`_require_admin`):
```python
def get_current_user(...) -> User: ...             # resolve from JWT email claim, 404 if missing
def get_active_membership(user=Depends(get_current_user)) -> OrganizationMembership: ...
                                                     # 403 if no active_organization_id / not a member
def require_permission(permission: str):
    def dep(membership=Depends(get_active_membership), user=Depends(get_current_user)):
        if user.is_superuser or membership.is_owner:
            return membership
        perms = membership.role.permissions
        if "*" in perms or permission in perms:
            return membership
        raise HTTPException(403, "Insufficient permissions.")
    return dep
```
`roles.py` and `organization.py` switch from the local name-based check to
`Depends(require_permission("manage_roles"))` / `Depends(require_permission("manage_members"))`.

Fixed permission vocabulary for v1: `manage_members`, `manage_roles`, `manage_organization`.

A second dependency, `get_org_membership`, resolves membership for an explicit `{org_id}` path
parameter instead of the caller's active org — added during implementation (see §0) for
endpoints like `GET /organizations/{org_id}` and the switch endpoint, where the org being acted
on isn't necessarily the one currently active. `require_org_permission`, built on top of it the
same way `require_permission` is built on `get_active_membership`, was added in Feature 2 for the
members endpoints (§0). A non-raising `user_has_permission`/`get_effective_permissions` pair
backs both — needed for `remove_member`'s self-removal bypass and for exposing the caller's
permission list via `GET /user` (§0, Feature 2 deviations).

### Registry structure now, vocabulary grows later

**v1 scope is deliberately just the three org-administration permissions above** —
`manage_members`, `manage_roles`, `manage_organization`. Product-feature permissions
(`document_upload`, `chat`, `voice_call`, `agent_create`, ...) are **not** being built now; that's
a separate, later decision (fine-grained product RBAC is a bigger scope than the 7 original
requirements). What we build now is the *structure* so that adding them later is purely additive
— append a line to a list, no migration, no schema change, no new endpoint.

That structure is **one registry file**, `ai_agent/core/permissions.py`, rather than raw strings
scattered across routers — even though today it only holds 3 entries:

```python
# ai_agent/core/permissions.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PermissionDef:
    key: str
    label: str
    description: str
    category: str

# v1: Organization-management permissions only.
# Future feature permissions (document_upload, chat, voice_call, agent_create, ...)
# get appended here later — same shape, same file, no schema change required.
PERMISSIONS = [
    PermissionDef("manage_members",      "Manage Members",      "Invite, remove, and change roles of members", "Organization"),
    PermissionDef("manage_roles",        "Manage Roles",        "Create, edit, and delete roles",               "Organization"),
    PermissionDef("manage_organization", "Manage Organization", "Rename org, edit org settings",                "Organization"),
]

VALID_KEYS = {p.key for p in PERMISSIONS}
```

The `category` field is included from day one even though every v1 entry is `"Organization"` —
it costs nothing now and is exactly what lets a future `"Documents"`/`"Chat"`/`"Agents"` category
slot in later without reshaping the data.

Why this stays one file rather than a `permissions` DB table, even once the list grows past a
handful of entries:
- Whoever adds a `document_upload`-gated endpoint still has to hardcode
  `Depends(require_permission("document_upload"))` at that call site regardless of where the
  definition lives — a DB table wouldn't remove that code-level source of truth, it would just
  add a second one that has to stay in sync with it.
- A permissions table earns its keep only if *admins* (not developers) need to invent new
  permission keys at runtime. "Add `document_upload` as a gated feature" is inherently a code
  change — someone has to write the `Depends(...)` call on that endpoint — so it's never
  admin-authored anyway.
- `Role.permissions` storage is unaffected either way — still a plain `list[str]` JSON column,
  now validated against `VALID_KEYS` in `RoleCreate`/`RoleUpdate` instead of a hardcoded
  `Literal[...]`.
- `"*"` (the system Admin role) keeps working unchanged — a wildcard role automatically gains
  every permission added to `PERMISSIONS` later, no backfill needed when a new feature ships.

This also motivates one more endpoint: **`GET /permissions`** (public/any authenticated user) —
returns the catalog grouped by `category` with `label`/`description`, so the role-creation UI in
the frontend renders its permission checkboxes dynamically from this one file instead of
hardcoding a second, driftable copy of the list.

## 6. Full API surface

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/register` | none | *(existing, internals updated)* signup → own org, `is_owner=True`, system Admin role |
| POST | `/login` | none | *(existing, unchanged)* |
| GET | `/user` | bearer | *(existing, updated)* now reads `active_organization_id` + membership role, plus `permissions: list[str]` (added in Feature 2, §0) |
| GET | `/organizations` | bearer | **NEW** — list every org the caller belongs to (id, name, role, is_owner, is_current) |
| POST | `/organizations` | bearer | *(existing, relaxed)* — "already has an org" 400 removed; creates org, caller becomes owner/Admin, becomes the new active org |
| GET | `/organizations/{org_id}` | bearer, member | **NEW** — org details |
| PATCH | `/organizations/{org_id}` | bearer, `manage_organization` | **NEW** (added post-launch, §0) — rename an organization |
| POST | `/organizations/{org_id}/switch` | bearer, member | **NEW** — set caller's `active_organization_id` |
| GET | `/organizations/{org_id}/members` | bearer, member | **NEW** — list accepted members: name, email, role, is_owner, joined_at |
| PATCH | `/organizations/{org_id}/members/{user_id}` | bearer, `manage_members` | **NEW** — change a member's role (403 if the target is the owner) |
| DELETE | `/organizations/{org_id}/members/{user_id}` | bearer, `manage_members` | **NEW** — remove a member (owner cannot be removed; self-removal = "leave org", blocked for the owner) |
| POST | `/organizations/{org_id}/invitations` | bearer, `manage_members` | **NEW** — invite by email + `role_id`; sends SES email |
| GET | `/organizations/{org_id}/invitations` | bearer, `manage_members` | **NEW** — list invites, filterable by status (pending/accepted/cancelled/expired) |
| DELETE | `/organizations/{org_id}/invitations/{invitation_id}` | bearer, `manage_members` | **NEW** — cancel a pending invite |
| POST | `/organizations/{org_id}/invitations/{invitation_id}/resend` | bearer, `manage_members` | **NEW** — re-send email, refresh expiry; also revives an EXPIRED invite (§0, Feature 3 deviations) |
| GET | `/invitations/{token}` | none (public) | **NEW** — resolve an invite for the accept-invite landing page (org name, inviter, role, whether the invitee already has an account) |
| POST | `/invitations/{token}/accept` | none / optional bearer | **NEW** — accept: signs up the user if new (returning `access_token`/`refresh_token` to auto-login them, §0), or attaches membership if the account already exists |
| GET/POST/PATCH/DELETE | `/roles` | bearer, `manage_roles` | *(existing, updated)* — permission-based gate instead of name check; `permissions` field added to create/update/response |
| GET | `/permissions` | bearer | **NEW** — the full permission catalog (key, label, description, category) from `core/permissions.py`, for the role-creation UI's checkbox list |

## 7. Business-logic flows

**Signup (`POST /register`)** — unchanged externally. Internally: create Keycloak identity →
create `Organization` → create system `Admin` role (`permissions=["*"]`) → create
`organization_memberships` row (`is_owner=True`) → set `active_organization_id`.

**Create additional org (`POST /organizations`)** — same as above minus Keycloak identity
creation; caller must already be authenticated. New org becomes the active one immediately
(standard "create workspace" UX).

**Invite (`POST /organizations/{org_id}/invitations`)**
1. Caller must hold `manage_members` in `org_id`.
2. Reject if `email` already has an *active* membership in `org_id` (409) or already has a
   *pending* invite there (409 — cancel or resend instead).
3. `role_id` must belong to `org_id`.
4. Generate `secrets.token_urlsafe(32)`, store its SHA-256 hash, `expires_at = now + 7d`.
5. Send SES email (styled like `identity.py`'s existing template) with
   `{FRONTEND_URL}/invitations/accept?token=...`.

**Accept (`POST /invitations/{token}/accept`)**
1. Look up by `token_hash`; 404/410 if missing/expired/already actioned.
2. If no local `User` row exists for the invited email: require `password`, `first_name`,
   `last_name` in the body → `create_identity_user(..., email_verified=True)` → create local
   `User`.
3. If a `User` row already exists: verify the caller's authenticated email matches the invited
   email (or, if unauthenticated, prompt them to log in first — `GET /invitations/{token}` tells
   the frontend which branch to render).
4. Create `organization_memberships` row (`role_id` from the invite, `is_owner=False`).
5. If the user had no `active_organization_id` yet, set it to this org; otherwise leave their
   current active org alone and let them switch manually.
6. Mark invitation `ACCEPTED`, stamp `accepted_at`/`accepted_user_id`.

**Switch (`POST /organizations/{org_id}/switch`)** — 403 if caller has no active membership in
`org_id`; otherwise sets `active_organization_id` and returns the new context (org, role).

**Remove member / leave org** — if the removed user's `active_organization_id == org_id`,
reassign it to another remaining membership, or `null` if that was their last org.

## 8. Example walkthrough (data flow)

A concrete run through the flows in §7, with example data, showing what moves where at each step.
Cast: Priya (starts fresh), Raj (Priya invites him), Acme Inc and Beta LLC (organizations).

**1. Signup → becomes Admin of her own org**
Priya signs up with `priya@acme.com` and org name "Acme Inc."
→ Creates organization "Acme Inc," a role called "Admin" with full permissions, and links Priya
to Acme Inc with that role, marked as *owner*.
→ Priya logs in and sees: **Acme Inc (Admin)** — her only org, automatically active.

**2. Priya invites Raj**
Priya opens "Members" in Acme Inc, picks a role for Raj (say "Editor" — a custom role she created
earlier with limited permissions), enters `raj@example.com`, clicks Invite.
→ Creates a pending invitation: *Acme Inc invited raj@example.com as Editor, sent by Priya,
expires in 7 days.* An email goes to Raj with a unique link.
→ Priya's "Invited Members" list shows: **raj@example.com — Editor — Pending**.

**3. Raj accepts (new user)**
Raj clicks the link. He has no account yet, so he's shown a signup form (name + password),
pre-filled with his email — no org-name field (see §10).
→ He submits it. His account is created, then immediately attached to Acme Inc with the Editor
role Priya picked.
→ Raj logs in and sees **Acme Inc (Editor)** as his active org.
→ On Priya's side, the invite flips **Pending → Accepted**, stamped with when Raj joined.

**4. Priya changes Raj's role later**
Priya opens Raj's row in the member list and changes his role from Editor to a custom role she
made, "Manager."
→ Next time Raj's permissions are checked, he has Manager-level access. No re-invite needed — a
live update to his existing membership.

**5. Priya cancels an invite she regrets**
Priya also invited `contractor@vendor.com` last week but the project fell through. She cancels it.
→ That invitation flips to **Cancelled** — the link in that email stops working. She can send a
fresh invite to the same address later if needed.

**6. Raj creates his own second organization**
Raj clicks "Create Organization," names it "Beta LLC."
→ Creates Beta LLC, makes Raj its owner/Admin (same as Priya's original signup flow), and
switches his active org to Beta LLC.
→ Raj now belongs to two orgs: **Acme Inc (Manager)** and **Beta LLC (Admin)**.

**7. Raj switches organizations**
Raj opens the org switcher and picks "Acme Inc."
→ His active org flips back to Acme Inc; everything he sees (agents, documents, members list) is
now scoped to Acme Inc, with his Manager role in effect there.
→ Switching back to Beta LLC makes him Admin there instead — same person, different hat,
depending on which org is currently active.

**The one idea underneath all of this**: every person has a list of orgs they belong to (each
with its own role), plus a pointer to *which one is currently active*. Inviting, accepting,
switching, and role changes are all just different ways of adding to, or updating, that list.

## 9. Edge cases explicitly handled

**Removing a member**
- Owner can't be removed or demoted by anyone (including other `manage_members` holders); owner
  also can't "leave" — must transfer ownership or delete the org (ownership transfer is a good v2
  addition, not in this pass).
- If the removed user's `active_organization_id` pointed at this org, it's reassigned to another
  of their remaining memberships, or set to `null` if this was their only org. A user with no
  active org sees a "join or create an organization" state on next login — expected, not a bug.
- The removed user's own resources (agents, documents, sessions — owned by `user_id`, not
  org-scoped per §10) are untouched; only the membership link is deleted.
- Self-removal ("leave organization") is allowed for non-owners, using the same
  `DELETE .../members/{user_id}` endpoint with `user_id == self`.

**Deleting a role**
- Blocked (400) if any *active membership* still references the role — existing behavior in
  today's `roles.py`, just re-pointed from checking `user_role` to checking
  `organization_memberships` (§3c).
- **Also blocked (400) if any *pending invitation* references the role** — an invite stores the
  `role_id` it will grant on acceptance; deleting that role out from under a pending invite would
  leave a dangling reference and break the accept flow later. Same guard pattern, same message
  style ("N pending invitation(s) use this role; cancel or reassign them before deleting"), just
  a second thing to check besides active members.
- System roles (`is_system=True`, i.e. the auto-created Admin role) can't be deleted or renamed at
  all — pre-existing protection in `roles.py`, now actually correct once the bootstrapped role is
  seeded with `is_system=True` (§2 bug fix).
- Even in the worst case — every role holding `manage_roles`/`manage_members` gets deleted — the
  org isn't locked out, because `is_owner` bypasses role permissions entirely (§3d). This is the
  actual reason `is_owner` exists as a separate flag rather than relying on role state alone.

**Invitations**
- Re-inviting after cancellation is allowed (partial unique index only blocks duplicate *pending*
  invites).
- Self-invite (admin invites their own email) → 400.
- Expired token on accept → 410 Gone, with a clear "ask the admin to resend" message.
- Race on Keycloak identity creation during accept (email already exists in Keycloak but not
  locally) — reuses `create_identity_user`'s existing 409 handling, surfaced as "an account
  already exists, please log in instead."

**Not designed yet (flagged, not silently skipped)**
- Deleting an entire organization — no endpoint proposed in §6. Comes up naturally once ownership
  transfer exists (an owner who wants to leave has to either transfer ownership or delete the
  org); treating both as a v2 addition together rather than half-building it now.

## 10. Explicitly out of scope for this pass

None of the 7 requirements mention whether org membership should also gate visibility of
agents/sessions/documents between members of the same org. Today those resources are scoped
purely by `user_id` ownership (`AIAgent.owner_id`, `Document.user_id`, etc.), with no
`organization_id` column at all. "Collaboration" suggests org members may eventually need to
share agents — but that's a materially bigger change (new FK columns on every resource table, ACL
checks in every existing router) and isn't implied by the requirements as given. Treat as a
separate follow-up design once actually needed.

## 11. Frontend impact (brief — `pb-ai-agent-41`, React + Vite)

- `AuthContext.tsx`'s `User` interface carries `organizationId`, `role`, `permissions`, and the
  full `organizations` list, plus `switchOrganization()`/`refreshUser()` methods.
- **Dedicated `/organization` page** (`src/pages/Organization.tsx`, linked from a new Sidebar
  item) — members list, invite form + invitations list, and role management (create/edit/delete
  with a permission-checkbox editor). Not a Settings tab — see §0's "Post-Feature-3 frontend
  restructuring" for why this moved out of Settings after the first pass.
- **`OrgSwitcher.tsx`** in the topbar next to the language selector — always visible, not buried
  in a menu; also hosts "create new organization."
- No auth flow changes — Keycloak login/signup stays the same; this is additive.

### Accept-invite is a separate page, not the existing signup page

The current signup page collects `org_name` because that user is *creating* a new org. An
invited user is *joining* one that already exists — asking them to name an org is wrong, and the
page needs invite context the plain signup form doesn't have. So this is a new route, not a mode
of the existing one:

- New route, e.g. `/invitations/accept?token=...` — the link in the invite email points here,
  never to `/signup`.
- On load it calls `GET /invitations/{token}` and shows a banner built from the response: which
  org, who invited them, and what role they're being given (e.g. "You've been invited to join
  **Acme Inc** by **Priya** as **Editor**").
- The form reuses the same fields as signup (first name, last name, password) **minus
  `org_name`**, and submits to `POST /invitations/{token}/accept` instead of `POST /register`.
- If the invited email already has an account, this page renders a "log in to join" variant
  instead of a signup form (branch determined by the `GET /invitations/{token}` response).
- An invalid/expired token shows an error state on this page — it never falls back to plain
  signup.

## 12. Suggested implementation sequencing

1. ✅ Migration + model changes (§4) — foundational, everything else depends on it.
2. ✅ `core/deps.py` permission layer (§5) — replaces the two existing routers' local checks.
3. ✅ Update `register`, `organization.py`, `roles.py` for the new membership model. Also shipped
   in this pass: `GET /organizations`, `GET /organizations/{org_id}`,
   `POST /organizations/{org_id}/switch`, `GET /permissions`, and the matching frontend (org
   switcher + create-org UI).
4. ✅ New `members.py` router (list/change-role/remove/leave). Also shipped: `GET /user` now
   returns `permissions`, and the matching frontend (Members tab in Settings).
5. ✅ New `invitations.py` router (token gen/hash, email send, accept logic — no separate
   `services/invitations.py`, see §0). Also shipped: `identity.py`'s `email_verified` param, and
   the matching frontend (invite/cancel/resend UI in the Members tab, plus the public
   `/invitations/accept` page).
6. ✅ Wired into `main.py`.

All three planned features are now implemented and awaiting live-DB verification (§0). See §0
for the full implementation-status breakdown and what was found/decided along the way.
