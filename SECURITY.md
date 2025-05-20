# Security Policy

## Reporting a Vulnerability

The Adamant team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

To report a security issue, please email [security-email](mailto:dr.saad.laouadi@gmail.com) with a description of the issue, the steps you took to create it, affected versions, and if known, mitigations. You should receive a response within 48 hours.

Please include the word "SECURITY" in the subject line of your email.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Model

Adamant is a tensor computation and automatic differentiation library designed for scientific computing and machine learning applications. As such, our security considerations include:

- Protection against memory safety issues (already provided by Rust's safety guarantees)
- Protecting against excessive resource consumption
- Ensuring correct numerical computation
- Safe handling of user-provided data and serialized tensors

Adamant is not intended to:
- Process untrusted serialized data
- Provide cryptographic operations
- Enforce access control mechanisms

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the vulnerability and determine its impact
2. Develop a fix and release it according to our severity assessment:
   - Critical: within 3 working days
   - High: within 7 working days
   - Medium/Low: within normal release schedule
3. Issue a security advisory via GitHub's security advisory feature

## Comments on Security

While Adamant is implemented in Rust, which provides strong memory safety guarantees, bugs can still occur. We are committed to addressing security issues promptly and transparently.

If you have suggestions for improving this security policy, please open an issue or pull request in our GitHub repository.
