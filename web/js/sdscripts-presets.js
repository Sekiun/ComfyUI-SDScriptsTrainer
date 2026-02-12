import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXTENSION_NAME = "comfyui.sdscripts_trainer.presets";
const NONE_PRESET = "(none)";
const TARGET_NODES = new Set(["SDScriptsDatasetConfig", "SDScriptsTrainParams"]);

let presetPayloadPromise = null;

const getWidget = (node, name) => (node.widgets || []).find((widget) => widget?.name === name);

const fetchPresetPayload = async () => {
    if (!presetPayloadPromise) {
        presetPayloadPromise = (async () => {
            const response = await api.fetchApi("/sdscripts_trainer/presets");
            if (!response.ok) {
                throw new Error(`Failed to load presets (${response.status})`);
            }
            return response.json();
        })().catch((error) => {
            console.warn("[SDScriptsTrainer] Failed to load xmlora presets:", error);
            presetPayloadPromise = null;
            return null;
        });
    }
    return presetPayloadPromise;
};

const clampNumber = (widget, value) => {
    let clamped = value;
    const min = Number(widget?.options?.min);
    const max = Number(widget?.options?.max);
    if (Number.isFinite(min)) {
        clamped = Math.max(clamped, min);
    }
    if (Number.isFinite(max)) {
        clamped = Math.min(clamped, max);
    }
    return clamped;
};

const setWidgetValue = (node, widgetName, nextRawValue) => {
    if (nextRawValue === undefined || nextRawValue === null) {
        return;
    }

    const widget = getWidget(node, widgetName);
    if (!widget) {
        return;
    }

    const previousValue = widget.value;
    let nextValue = nextRawValue;

    if (typeof previousValue === "boolean") {
        nextValue = Boolean(nextRawValue);
    } else if (typeof previousValue === "number") {
        const parsed = Number(nextRawValue);
        if (!Number.isFinite(parsed)) {
            return;
        }
        const clamped = clampNumber(widget, parsed);
        nextValue = Number.isInteger(previousValue) ? Math.round(clamped) : clamped;
    } else if (widget.options?.values && Array.isArray(widget.options.values)) {
        const values = widget.options.values;
        const direct = values.find((value) => value === nextRawValue);
        if (direct !== undefined) {
            nextValue = direct;
        } else {
            const lowerTarget = String(nextRawValue).toLowerCase();
            const looseMatch = values.find((value) => String(value).toLowerCase() === lowerTarget);
            if (looseMatch === undefined) {
                return;
            }
            nextValue = looseMatch;
        }
    } else {
        nextValue = String(nextRawValue);
    }

    if (nextValue === previousValue) {
        return;
    }

    widget.value = nextValue;
    if (widget.inputEl) {
        widget.inputEl.value = String(nextValue);
    }
    node.onWidgetChanged?.call(node, widget, nextValue, previousValue, widget);
};

const applyPreset = async (node, nodeClass, presetName) => {
    if (!presetName || presetName === NONE_PRESET) {
        return;
    }

    const payload = await fetchPresetPayload();
    if (!payload) {
        return;
    }

    const groupKey = nodeClass === "SDScriptsDatasetConfig" ? "dataset" : "train";
    const values = payload?.[groupKey]?.[presetName];
    if (!values || typeof values !== "object") {
        return;
    }

    Object.entries(values).forEach(([widgetName, value]) => {
        setWidgetValue(node, widgetName, value);
    });

    app.graph?.setDirtyCanvas?.(true, true);
};

const bindPresetCallback = (node, nodeClass) => {
    if (node.__sdscriptsPresetBound) {
        return;
    }

    const presetWidget = getWidget(node, "preset_xmlora");
    if (!presetWidget) {
        return;
    }

    const originalCallback = presetWidget.callback;
    presetWidget.callback = function presetChanged(value) {
        if (typeof originalCallback === "function") {
            originalCallback.apply(this, arguments);
        }
        const selected = typeof value === "string" ? value : presetWidget.value;
        void applyPreset(node, nodeClass, selected);
    };

    node.__sdscriptsPresetBound = true;
};

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeClass = nodeData?.name;
        if (!TARGET_NODES.has(nodeClass)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedSDScriptsPresets() {
            originalOnNodeCreated?.apply(this, arguments);
            bindPresetCallback(this, nodeClass);
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigureSDScriptsPresets(info) {
            originalOnConfigure?.apply(this, arguments);
            bindPresetCallback(this, nodeClass);
        };
    },
});
