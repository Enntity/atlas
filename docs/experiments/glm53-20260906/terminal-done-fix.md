# Terminal Done flush correction

## Observed v12 defect

The second row of
`/tmp/atlas-glm53-phase5-20260907.gu145h/v12-stream-stop-probes.jsonl`
records ordinary content `iet` at 2.997 seconds and additional content
`The lantern` at 3.1446 seconds, after the semantic watchdog and scheduler
completion. Logs place the guard at 00:37:43.828651 and Done at
00:37:43.901547: scheduler cancellation is prompt, but Done unconditionally
flushes detector and sanitizer buffers. The Token entry gate cannot cover Done.

## Minimal test-first plan

1. Extract the actual Done delta-finalization core from metrics/refund/dump
   side effects. Keep the existing pending-output flush body intact behind a
   callback; use real StreamState and sanitizer buffers in CPU tests.
2. Observe failing tests for a pre-existing semantic guard and a guard raised
   during finalization. Check actual Finish usage/reason/IDs, not only a bool.
3. For `guard_stop`, `tool_loop_capped`, or a generic terminal stop flag without
   a genuine client stop match, skip pending-output flushing. Recheck after a
   flush in case a tool handler trips a guard. Discard all buffered generated
   deltas and pending IDs; emit no new refusal classification after a guard.
4. Preserve genuine explicit-stop pre-stop bytes, normal EOS tails, detector
   history and the existing finish-reason precedence. Usage remains scheduler
   authority. IDs need not sum to completion usage: EOS may not be streamed,
   and guarded/rejected buffered tokens must not reappear on Finish.
5. Run CPU regression, neighboring chat-stream and formatting checks; freeze
   for independent review. The deployment owner rebuilds and repeats the
   bounded live stop probes separately.

No watchdog threshold, kernel, dtype, cancellation race, server admission or
finish-reason policy changes. No GPU or remote operations in this work item.

## CPU evidence

The extracted production finalizer with its original unconditional flush failed
four regression tests and passed the normal-EOS/explicit-stop control. Failures
included the exact `The lantern` sanitizer tail plus refusal, tool-flush entry,
all generated delta kinds after a flush-time guard, and nonempty generated output
alongside otherwise-correct finish precedence. The guarded implementation passes
all five tests. The complete `chat_stream` filter passes 55/55, and the adjacent
`api::stream_guards` filter passes 7/7. Scoped rustfmt and diff checks pass.

The moved pending-output flush body is byte-identical to v12. Metrics, refund,
dump data and scheduler-derived usage arithmetic are unchanged. Independent
read-only review found no blocker. Live v13 stop probes remain the deployment
owner's gate; CPU tests alone do not establish deployed behavior.
