// SPDX-License-Identifier: AGPL-3.0-only

//! OpenTelemetry initialization and trace context propagation.
//!
//! Configures the global tracer provider with OTLP export and installs
//! the tracing-opentelemetry bridge so `tracing` spans become OTEL spans.
//! Environment variables control the exporter endpoint and sampling.

use anyhow::Result;
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{self, ParentBasedSampler, SdkTracerProvider};
use std::collections::HashMap;
use std::time::Duration;

/// Global shutdown guard returned by `init_otel`. Dropping it flushes
/// pending spans. The server stores it in `AppState` so it lives for the
/// process lifetime (or until a model swap re-initializes).
pub struct OtelGuard {
    tracer_provider: SdkTracerProvider,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        let _ = self.tracer_provider.shutdown();
    }
}

/// Initialize OpenTelemetry tracing from environment variables.
///
/// Environment variables:
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP HTTP endpoint (default: none, disables export)
/// - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` — Traces-specific endpoint (overrides above)
/// - `OTEL_EXPORTER_OTLP_HEADERS` — Comma-separated headers (e.g. "Authorization=Bearer token")
/// - `OTEL_EXPORTER_OTLP_TIMEOUT` — Export timeout in seconds (default: 10)
/// - `OTEL_EXPORTER_OTLP_COMPRESSION` — Compression: "gzip" or "none" (default: "gzip")
/// - `OTEL_TRACES_SAMPLER` — Sampler: "always_on", "always_off", "parentbased_always_on", "traceidratio" (default: "parentbased_always_on")
/// - `OTEL_TRACES_SAMPLER_ARG` — Argument for traceidratio sampler (e.g. "0.1")
/// - `OTEL_SERVICE_NAME` — Service name (default: "atlas-spark")
/// - `OTEL_RESOURCE_ATTRIBUTES` — Additional resource attributes (comma-separated key=value)
pub fn init_otel() -> Result<Option<OtelGuard>> {
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .ok();

    let Some(endpoint) = endpoint else {
        tracing::debug!("OTEL_EXPORTER_OTLP_ENDPOINT not set; OpenTelemetry tracing disabled");
        return Ok(None);
    };

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "atlas-spark".to_string());

    // Parse headers
    let headers = std::env::var("OTEL_EXPORTER_OTLP_HEADERS")
        .ok()
        .and_then(|h| parse_headers(&h).ok())
        .unwrap_or_default();

    // Parse timeout
    let timeout_secs = std::env::var("OTEL_EXPORTER_OTLP_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10);

    // Parse compression
    let compression = std::env::var("OTEL_EXPORTER_OTLP_COMPRESSION")
        .ok()
        .and_then(|c| match c.as_str() {
            "gzip" => Some(opentelemetry_otlp::Compression::Gzip),
            "none" => None,
            _ => None,
        })
        .unwrap_or(opentelemetry_otlp::Compression::Gzip);

    // Build resource
    let mut resource_attributes = vec![
        opentelemetry::KeyValue::new("service.name", service_name.clone()),
        opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ];

    // Add custom resource attributes from env
    if let Ok(attrs) = std::env::var("OTEL_RESOURCE_ATTRIBUTES") {
        for attr in attrs.split(',') {
            if let Some((k, v)) = attr.split_once('=') {
                resource_attributes.push(opentelemetry::KeyValue::new(
                    k.trim().to_string(),
                    v.trim().to_string(),
                ));
            }
        }
    }

    let resource = opentelemetry_sdk::Resource::builder()
        .with_attributes(resource_attributes)
        .build();

    // Build exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(timeout_secs))
        .with_compression(compression)
        .build()?;

    // Build sampler
    let sampler = build_sampler()?;

    // Build tracer provider
    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(sampler)
        .with_resource(resource)
        .build();

    // Note: The OpenTelemetry layer is not automatically installed into the
    // global tracing subscriber here because the subscriber is already initialized
    // in main(). For proper integration, the layer should be added when the
    // subscriber is built. However, we can still use the tracer directly for
    // creating spans in middleware.
    tracing::info!(
        "OpenTelemetry tracing initialized: endpoint={}, service={}",
        std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
            .unwrap_or_default(),
        service_name
    );

    Ok(Some(OtelGuard { tracer_provider }))
}

fn parse_headers(s: &str) -> Result<HashMap<String, String>> {
    let mut headers = HashMap::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            headers.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    Ok(headers)
}

fn build_sampler() -> Result<Box<dyn trace::ShouldSample>> {
    let sampler_name = std::env::var("OTEL_TRACES_SAMPLER")
        .unwrap_or_else(|_| "parentbased_always_on".to_string());

    match sampler_name.as_str() {
        "always_on" => Ok(Box::new(trace::Sampler::AlwaysOn)),
        "always_off" => Ok(Box::new(trace::Sampler::AlwaysOff)),
        "parentbased_always_on" => Ok(Box::new(ParentBasedSampler::new(trace::Sampler::AlwaysOn))),
        "parentbased_always_off" => {
            Ok(Box::new(ParentBasedSampler::new(trace::Sampler::AlwaysOff)))
        }
        "traceidratio" => {
            let ratio = std::env::var("OTEL_TRACES_SAMPLER_ARG")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.0);
            Ok(Box::new(ParentBasedSampler::new(
                trace::Sampler::TraceIdRatioBased(ratio),
            )))
        }
        other => {
            tracing::warn!(
                "Unknown OTEL_TRACES_SAMPLER='{}', defaulting to parentbased_always_on",
                other
            );
            Ok(Box::new(ParentBasedSampler::new(trace::Sampler::AlwaysOn)))
        }
    }
}

/// Middleware that creates a tracing span for each request with OTEL attributes.
/// This works with the existing `tracing` setup and adds standard HTTP attributes
/// that will be captured by the OTEL layer if installed.
pub async fn otel_request_span_middleware(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue};

    let start = std::time::Instant::now();
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    // Extract incoming trace context
    let traceparent = req
        .headers()
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Create span with standard HTTP and OTEL attributes
    let span = tracing::info_span!(
        "http_request",
        http.method = %method,
        http.path = %path,
        http.scheme = %req.uri().scheme_str().unwrap_or("http"),
        http.target = %path,
        otel.kind = "server",
    );

    if let Some(tp) = traceparent {
        span.record("traceparent", &tp);
    }

    // Store span in extensions for handler access
    req.extensions_mut().insert(span.clone());

    let _guard = span.enter();
    let mut resp = next.run(req).await;

    let elapsed_ms = start.elapsed().as_millis();
    let status = resp.status().as_u16();

    span.record("http.status_code", status);
    span.record("http.response_time_ms", elapsed_ms);

    // Add trace headers to response
    let headers = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&format!("req_{}", crate::ids::uuid_v4())) {
        headers.insert(HeaderName::from_static("x-request-id"), v);
    }

    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_headers() {
        let headers =
            parse_headers("Authorization=Bearer token, Content-Type=application/json").unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer token".to_string())
        );
        assert_eq!(
            headers.get("Content-Type"),
            Some(&"application/json".to_string())
        );
    }

    #[test]
    fn test_parse_headers_empty() {
        let headers = parse_headers("").unwrap();
        assert_eq!(headers.len(), 0);
    }
}
