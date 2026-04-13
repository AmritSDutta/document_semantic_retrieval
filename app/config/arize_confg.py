from arize.otel import register
from app.config.Settings import get_settings

# Set up the tracer provider
if get_settings().ARIZE_ENABLE_TRACING == "true":
    tracer_provider = register(
        space_id=get_settings().ARIZE_SPACE_ID,
        api_key=get_settings().ARIZE_API_KEY,
        project_name="DSR",
    )

    tracer = tracer_provider.get_tracer(__name__)
else:
    # Create a dummy tracer with a .chain no-op decorator
    class NoOpTracer:
        def chain(self, func):
            return func

        def start_as_current_span(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()


    tracer = NoOpTracer()
