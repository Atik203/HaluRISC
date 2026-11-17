# Security Policy

## Supported versions

Security fixes are applied to the `main` branch. Older commits and archived
branches are not supported.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub:

1. Open the repository on GitHub.
2. Go to **Security** and choose **Report a vulnerability**.
3. Describe the issue, the affected component, and the steps to reproduce it.

If private reporting is unavailable, email the corresponding author:

**mrahaman2310298@bscse.uiu.ac.bd**

Include:

- The affected file, route, or dependency.
- A minimal reproduction.
- The impact you believe the issue has.
- Any suggested fix.

## What to expect

- Acknowledgement within five working days.
- A short assessment and a fix plan.
- Credit in the release notes if you want it.

## Scope

In scope:

- The FastAPI service in `src/api/`.
- The Next.js application in `web/`.
- Dependency vulnerabilities in `requirements.txt` and `web/package.json`.
- Accidental exposure of credentials, datasets, or personal data.

Out of scope:

- Denial of service against the public demo.
- Issues that require a compromised host or browser.
- Reports about model accuracy or scientific claims; use the normal issue tracker.

## Safe harbour

Good-faith research that follows this policy will not be treated as a violation
of the project licence or terms. Do not access data that is not yours, and do
not degrade the public demo for other users.
