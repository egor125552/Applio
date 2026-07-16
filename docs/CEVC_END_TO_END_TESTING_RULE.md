# Mandatory CEVC end-to-end validation

A user-facing CEVC workflow is not complete after unit tests, import tests, or isolated function tests alone.

Before presenting a workflow to the user, the branch must pass an end-to-end test that follows the same path as the user:

1. Prepare representative input files.
2. Start the real Gradio application.
3. Operate the relevant controls with browser automation.
4. Run the production processing or training path.
5. For training, execute real forward, backward and optimizer operations for at least 20 optimizer steps.
6. Verify persistent checkpoints, summary JSON, history JSON and files returned through Gradio.
7. Reproduce Google-Drive-like external paths or symlinks.
8. Fix every discovered error and rerun the complete path from the beginning.

Compile-only, import-only and mocked-success checks do not satisfy this rule.

Small public audio fixtures may be used to keep CI practical, but production modules and production control flow must be exercised. Any synthetic transformation used only to distinguish engineering fixture classes must be identified as a test fixture and must not be treated as acoustic evidence.
