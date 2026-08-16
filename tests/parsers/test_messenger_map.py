"""Tests for the Symfony Messenger contract parser."""
from scripts.parsers import PROJECT_ROOT, messenger_map


# --- Real-repo shape / known wiring ----------------------------------------

def test_parse_returns_expected_shape():
    result = messenger_map.parse(PROJECT_ROOT)
    assert "messages" in result
    assert "orphans" in result
    assert "stats" in result
    for key in ("total_messages", "total_handlers", "total_producer_sites",
                "unhandled", "undispatched", "unresolved_dispatch"):
        assert key in result["stats"], f"missing stat {key}"
    assert "unhandled" in result["orphans"]
    assert "undispatched" in result["orphans"]


def test_finds_known_handlers():
    result = messenger_map.parse(PROJECT_ROOT)
    # The repo ships ~14 #[AsMessageHandler] classes.
    assert result["stats"]["total_handlers"] >= 5, result["stats"]


def test_grade_session_message_maps_to_its_handler():
    result = messenger_map.parse(PROJECT_ROOT)
    hit = next(
        (e for fqcn, e in result["messages"].items() if e["short"] == "GradeSessionMessage"),
        None,
    )
    assert hit is not None, "expected GradeSessionMessage to be discovered"
    handler_classes = [h["handler_class"] for h in hit["handlers"]]
    assert any(c.endswith("GradeSessionHandler") for c in handler_classes), handler_classes
    # It is dispatched somewhere in the app.
    assert hit["producers"], "expected at least one GradeSessionMessage dispatch site"


def test_event_dispatcher_calls_are_not_tracked_as_messages():
    """``->dispatch(new App\\Event\\SessionGradedEvent())`` must not appear as a
    messenger message — it goes through EventDispatcher, not the message bus."""
    result = messenger_map.parse(PROJECT_ROOT)
    shorts = {e["short"] for e in result["messages"].values()}
    assert "SessionGradedEvent" not in shorts


def test_summary_returns_short_string():
    s = messenger_map.summary(PROJECT_ROOT)
    assert isinstance(s, str)
    assert len(s) < 300


# --- Synthetic parsing (deterministic, exercises each detection branch) -----

_CLASS_LEVEL_HANDLER = """<?php
namespace App\\MessageHandler;

use App\\Message\\FooMessage;
use Symfony\\Component\\Messenger\\Attribute\\AsMessageHandler;

#[AsMessageHandler]
class FooHandler
{
    public function __invoke(FooMessage $message): void
    {
    }
}
"""

_METHOD_LEVEL_HANDLER = """<?php
namespace App\\MessageHandler;

use App\\Message\\BarMessage;
use Symfony\\Component\\Messenger\\Attribute\\AsMessageHandler;

class BarHandler
{
    #[AsMessageHandler]
    public function handleBar(BarMessage $message): void
    {
    }
}
"""

_HANDLER_WITH_BUS_ARG = """<?php
namespace App\\MessageHandler;

use App\\Message\\BazMessage;
use Symfony\\Component\\Messenger\\Attribute\\AsMessageHandler;

#[AsMessageHandler(bus: 'messenger.bus.scenario_generation')]
class BazHandler
{
    public function __invoke(BazMessage $message): void
    {
    }
}
"""

_PRODUCER = """<?php
namespace App\\Controller;

use App\\Message\\FooMessage;
use App\\Event\\SomethingHappenedEvent;

class SomeController
{
    public function act(): void
    {
        $this->bus->dispatch(new FooMessage(1, 2));
        $this->eventDispatcher->dispatch(new SomethingHappenedEvent());
        $this->bus->dispatchAfterCurrentBus(new FooMessage(3));
    }
}
"""


def test_class_level_invoke_handler():
    handlers, _ = messenger_map._parse_content(_CLASS_LEVEL_HANDLER, "src/MessageHandler/FooHandler.php")
    assert len(handlers) == 1
    h = handlers[0]
    assert h["message_fqcn"] == "App\\Message\\FooMessage"
    assert h["handler_class"] == "App\\MessageHandler\\FooHandler"
    assert h["handler_method"] == "__invoke"


def test_method_level_handler():
    handlers, _ = messenger_map._parse_content(_METHOD_LEVEL_HANDLER, "src/MessageHandler/BarHandler.php")
    assert len(handlers) == 1
    h = handlers[0]
    assert h["message_fqcn"] == "App\\Message\\BarMessage"
    assert h["handler_method"] == "handleBar"


def test_handler_with_bus_arg_still_resolves_message():
    handlers, _ = messenger_map._parse_content(_HANDLER_WITH_BUS_ARG, "src/MessageHandler/BazHandler.php")
    assert len(handlers) == 1
    assert handlers[0]["message_fqcn"] == "App\\Message\\BazMessage"


def test_producer_detection_and_event_exclusion():
    _, producers = messenger_map._parse_content(_PRODUCER, "src/Controller/SomeController.php")
    foo = [p for p in producers if p["message_short"] == "FooMessage"]
    assert len(foo) == 2, "both dispatch + dispatchAfterCurrentBus should be found"
    assert {p["via"] for p in foo} == {"dispatch", "dispatchAfterCurrentBus"}
    # The event is still *parsed* as a producer row here; the App\\Message\\
    # filter that drops it happens in parse(), verified below.
    event = [p for p in producers if p["message_short"] == "SomethingHappenedEvent"]
    assert event and event[0]["message_fqcn"] == "App\\Event\\SomethingHappenedEvent"


def test_parse_level_filter_drops_event_but_keeps_message():
    """End-to-end over synthetic files: the event producer is filtered out at
    the aggregation layer, the message producer is kept and marked unhandled."""
    # Simulate aggregation the way parse() does, without touching the repo.
    handlers, producers = messenger_map._parse_content(_PRODUCER, "src/Controller/SomeController.php")
    handled = {h["message_fqcn"] for h in handlers}
    kept = [
        p for p in producers
        if p["message_fqcn"].startswith("App\\Message\\") or p["message_fqcn"] in handled
    ]
    kept_shorts = {p["message_short"] for p in kept}
    assert "FooMessage" in kept_shorts
    assert "SomethingHappenedEvent" not in kept_shorts


# --- Regression cases for the adversarial-review findings -------------------

def _handlers(text):
    return messenger_map._parse_content(text, "src/MessageHandler/H.php")[0]


def _producers(text):
    return messenger_map._parse_content(text, "src/X.php")[1]


def test_c1_grouped_use_resolves_message_fqcn():
    """Grouped `use App\\Message\\{Foo, Bar};` must resolve, not fall back to
    the handler's own namespace (which produced two false orphans)."""
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\{FooMessage, BarMessage};
use Symfony\\Component\\Messenger\\Attribute\\AsMessageHandler;
#[AsMessageHandler]
class FooHandler { public function __invoke(FooMessage $m): void {} }
"""
    hs = _handlers(text)
    assert len(hs) == 1
    assert hs[0]["message_fqcn"] == "App\\Message\\FooMessage"


def test_h2_union_typed_handler_emits_a_row_per_member():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
use App\\Message\\BarMessage;
#[AsMessageHandler]
class DualHandler { public function __invoke(FooMessage|BarMessage $m): void {} }
"""
    fqcns = {h["message_fqcn"] for h in _handlers(text)}
    assert fqcns == {"App\\Message\\FooMessage", "App\\Message\\BarMessage"}


def test_h3_docblock_mentioning_class_does_not_fabricate_handler_name():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
/**
 * This class grades sessions after each call.
 */
#[AsMessageHandler]
final class GradeHandler { public function __invoke(FooMessage $m): void {} }
"""
    hs = _handlers(text)
    assert len(hs) == 1
    assert hs[0]["handler_class"] == "App\\MessageHandler\\GradeHandler"


def test_h5_commented_out_attribute_is_not_a_handler():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
// #[AsMessageHandler]
class FooHandler { public function __invoke(FooMessage $m): void {} }
"""
    assert _handlers(text) == []


def test_h4_envelope_wrapped_dispatch_unwraps_to_inner_message():
    text = """<?php
namespace App\\Controller;
use App\\Message\\FooMessage;
class C { public function a(): void {
    $this->bus->dispatch(new Envelope(new FooMessage(1), [new DelayStamp(5000)]));
} }
"""
    ps = _producers(text)
    assert any(p["message_fqcn"] == "App\\Message\\FooMessage" for p in ps)
    assert all(p["message_short"] != "Envelope" for p in ps)


def test_h6_relative_qualified_name_resolves_via_import():
    text = """<?php
namespace App\\Controller;
use App\\Message;
class C { public function a(): void {
    $this->bus->dispatch(new Message\\FooMessage(1));
} }
"""
    ps = _producers(text)
    assert any(p["message_fqcn"] == "App\\Message\\FooMessage" for p in ps)


def test_m7_grouped_attribute_list_still_detects_handler():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
#[AsMessageHandler, SomeOtherAttr]
class FooHandler { public function __invoke(FooMessage $m): void {} }
"""
    assert len(_handlers(text)) == 1


def test_m8_two_handler_classes_in_one_file_are_not_conflated():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\AMessage;
use App\\Message\\BMessage;
#[AsMessageHandler]
class AHandler { public function __invoke(AMessage $m): void {} }
#[AsMessageHandler]
class BHandler { public function __invoke(BMessage $m): void {} }
"""
    by_msg = {h["message_short"]: h["handler_class"] for h in _handlers(text)}
    assert by_msg["AMessage"].endswith("AHandler")
    assert by_msg["BMessage"].endswith("BHandler")


def test_m9_parameter_attribute_does_not_hide_the_message_type():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
#[AsMessageHandler]
class FooHandler {
    public function __invoke(#[SensitiveParameter(redact: true)] FooMessage $m): void {}
}
"""
    hs = _handlers(text)
    assert len(hs) == 1
    assert hs[0]["message_fqcn"] == "App\\Message\\FooMessage"


def test_m12_method_argument_reports_the_real_handler_method():
    text = """<?php
namespace App\\MessageHandler;
use App\\Message\\FooMessage;
#[AsMessageHandler(handles: FooMessage::class, method: 'handleIt')]
class FooHandler { public function handleIt(FooMessage $m): void {} }
"""
    hs = _handlers(text)
    assert len(hs) == 1
    assert hs[0]["handler_method"] == "handleIt"
    assert hs[0]["message_fqcn"] == "App\\Message\\FooMessage"


def test_parse_aggregation_and_orphans_on_a_temp_project(tmp_path):
    """End-to-end over parse() (not a re-implemented filter): a handled+
    dispatched message, an unhandled-but-dispatched message, and an event
    that must be excluded."""
    src = tmp_path / "src"
    (src / "MessageHandler").mkdir(parents=True)
    (src / "Controller").mkdir(parents=True)
    (src / "MessageHandler" / "FooHandler.php").write_text(
        "<?php\nnamespace App\\MessageHandler;\n"
        "use App\\Message\\FooMessage;\n"
        "#[AsMessageHandler]\n"
        "class FooHandler { public function __invoke(FooMessage $m): void {} }\n",
        encoding="utf-8",
    )
    (src / "Controller" / "C.php").write_text(
        "<?php\nnamespace App\\Controller;\n"
        "use App\\Message\\FooMessage;\n"
        "use App\\Message\\LonelyMessage;\n"
        "use App\\Event\\SessionGradedEvent;\n"
        "class C { public function a(): void {\n"
        "    $this->bus->dispatch(new FooMessage(1));\n"
        "    $this->bus->dispatch(new LonelyMessage());\n"
        "    $this->events->dispatch(new SessionGradedEvent());\n"
        "} }\n",
        encoding="utf-8",
    )
    result = messenger_map.parse(tmp_path)
    shorts = {e["short"] for e in result["messages"].values()}
    assert "FooMessage" in shorts
    assert "LonelyMessage" in shorts
    assert "SessionGradedEvent" not in shorts, "EventDispatcher call leaked in"
    unhandled_shorts = {s.split("\\")[-1] for s in result["orphans"]["unhandled"]}
    assert "LonelyMessage" in unhandled_shorts
    assert "FooMessage" not in unhandled_shorts
