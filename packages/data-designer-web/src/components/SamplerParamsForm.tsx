/**
 * Dynamic parameter forms for each sampler type.
 *
 * Each sampler type has different required/optional fields. This component
 * renders the right form fields based on the selected sampler type and
 * manages a flat Record<string,unknown> of parameter values.
 */

interface Props {
  samplerType: string;
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="label-text">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-500 mt-1">{hint}</p>}
    </div>
  );
}

export default function SamplerParamsForm({ samplerType, params, onChange }: Props) {
  const set = (key: string, value: unknown) =>
    onChange({ ...params, [key]: value });

  const str = (key: string, fallback = "") =>
    (params[key] as string) ?? fallback;
  const num = (key: string, fallback?: number) =>
    params[key] != null ? Number(params[key]) : fallback;
  const bool = (key: string, fallback = false) =>
    (params[key] as boolean) ?? fallback;

  switch (samplerType) {
    // ---- No required params (all optional) ----
    case "uuid":
      return (
        <div className="grid grid-cols-3 gap-4">
          <Field label="Prefix" hint="Prepended to UUID">
            <input
              className="input-field"
              value={str("prefix")}
              onChange={(e) => set("prefix", e.target.value || undefined)}
              placeholder="e.g. user-"
            />
          </Field>
          <Field label="Short Form">
            <label className="flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                checked={bool("short_form")}
                onChange={(e) => set("short_form", e.target.checked)}
                className="accent-[#76b900]"
              />
              <span className="text-sm text-gray-300">Truncate to 8 chars</span>
            </label>
          </Field>
          <Field label="Uppercase">
            <label className="flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                checked={bool("uppercase")}
                onChange={(e) => set("uppercase", e.target.checked)}
                className="accent-[#76b900]"
              />
              <span className="text-sm text-gray-300">Uppercase letters</span>
            </label>
          </Field>
        </div>
      );

    case "category":
      return (
        <div className="space-y-4">
          <Field label="Values (comma-separated)" hint="At least one value required">
            <input
              className="input-field"
              value={
                Array.isArray(params.values)
                  ? (params.values as string[]).join(", ")
                  : str("values")
              }
              onChange={(e) =>
                set(
                  "values",
                  e.target.value
                    .split(",")
                    .map((v: string) => v.trim())
                    .filter(Boolean)
                )
              }
              placeholder="value1, value2, value3"
            />
          </Field>
        </div>
      );

    case "subcategory":
      return (
        <div className="space-y-4">
          <Field label="Parent Category Column" hint="Name of the category column this depends on">
            <input
              className="input-field"
              value={str("category")}
              onChange={(e) => set("category", e.target.value)}
              placeholder="e.g. department"
            />
          </Field>
          <Field label="Values (JSON)" hint='e.g. {"Sales": ["Rep", "Manager"], "Eng": ["Junior", "Senior"]}'>
            <textarea
              className="input-field font-mono text-xs min-h-[80px]"
              value={
                typeof params.values === "object" && !Array.isArray(params.values)
                  ? JSON.stringify(params.values, null, 2)
                  : str("values")
              }
              onChange={(e) => {
                try {
                  set("values", JSON.parse(e.target.value));
                } catch {
                  set("values", e.target.value);
                }
              }}
              placeholder='{"category_value": ["sub1", "sub2"]}'
            />
          </Field>
        </div>
      );

    case "datetime":
      return (
        <div className="grid grid-cols-3 gap-4">
          <Field label="Start" hint="e.g. 2024-01-01">
            <input
              className="input-field"
              value={str("start")}
              onChange={(e) => set("start", e.target.value)}
              placeholder="2024-01-01"
            />
          </Field>
          <Field label="End" hint="e.g. 2024-12-31">
            <input
              className="input-field"
              value={str("end")}
              onChange={(e) => set("end", e.target.value)}
              placeholder="2024-12-31"
            />
          </Field>
          <Field label="Unit">
            <select
              className="select-field"
              value={str("unit", "D")}
              onChange={(e) => set("unit", e.target.value)}
            >
              <option value="Y">Year</option>
              <option value="M">Month</option>
              <option value="D">Day</option>
              <option value="h">Hour</option>
              <option value="m">Minute</option>
              <option value="s">Second</option>
            </select>
          </Field>
        </div>
      );

    case "timedelta":
      return (
        <div className="grid grid-cols-2 gap-4">
          <Field label="Min Delta" hint="Non-negative integer">
            <input
              className="input-field"
              type="number"
              min={0}
              value={num("dt_min") ?? ""}
              onChange={(e) => set("dt_min", parseInt(e.target.value) || 0)}
              placeholder="0"
            />
          </Field>
          <Field label="Max Delta" hint="Must be > min">
            <input
              className="input-field"
              type="number"
              min={1}
              value={num("dt_max") ?? ""}
              onChange={(e) => set("dt_max", parseInt(e.target.value) || 1)}
              placeholder="30"
            />
          </Field>
          <Field label="Reference Column" hint="Existing datetime column">
            <input
              className="input-field"
              value={str("reference_column_name")}
              onChange={(e) => set("reference_column_name", e.target.value)}
              placeholder="e.g. order_date"
            />
          </Field>
          <Field label="Unit">
            <select
              className="select-field"
              value={str("unit", "D")}
              onChange={(e) => set("unit", e.target.value)}
            >
              <option value="D">Day</option>
              <option value="h">Hour</option>
              <option value="m">Minute</option>
              <option value="s">Second</option>
            </select>
          </Field>
        </div>
      );

    case "uniform":
      return (
        <div className="grid grid-cols-3 gap-4">
          <Field label="Low">
            <input
              className="input-field"
              type="number"
              step="any"
              value={num("low") ?? ""}
              onChange={(e) => set("low", parseFloat(e.target.value))}
              placeholder="0.0"
            />
          </Field>
          <Field label="High">
            <input
              className="input-field"
              type="number"
              step="any"
              value={num("high") ?? ""}
              onChange={(e) => set("high", parseFloat(e.target.value))}
              placeholder="1.0"
            />
          </Field>
          <Field label="Decimal Places" hint="Optional">
            <input
              className="input-field"
              type="number"
              min={0}
              value={num("decimal_places") ?? ""}
              onChange={(e) =>
                set("decimal_places", e.target.value ? parseInt(e.target.value) : undefined)
              }
              placeholder="auto"
            />
          </Field>
        </div>
      );

    case "gaussian":
      return (
        <div className="grid grid-cols-3 gap-4">
          <Field label="Mean">
            <input
              className="input-field"
              type="number"
              step="any"
              value={num("mean") ?? ""}
              onChange={(e) => set("mean", parseFloat(e.target.value))}
              placeholder="0.0"
            />
          </Field>
          <Field label="Std Dev">
            <input
              className="input-field"
              type="number"
              step="any"
              value={num("stddev") ?? ""}
              onChange={(e) => set("stddev", parseFloat(e.target.value))}
              placeholder="1.0"
            />
          </Field>
          <Field label="Decimal Places" hint="Optional">
            <input
              className="input-field"
              type="number"
              min={0}
              value={num("decimal_places") ?? ""}
              onChange={(e) =>
                set("decimal_places", e.target.value ? parseInt(e.target.value) : undefined)
              }
              placeholder="auto"
            />
          </Field>
        </div>
      );

    case "bernoulli":
      return (
        <div className="max-w-xs">
          <Field label="Probability (p)" hint="0.0 to 1.0">
            <input
              className="input-field"
              type="number"
              step="0.01"
              min={0}
              max={1}
              value={num("p") ?? ""}
              onChange={(e) => set("p", parseFloat(e.target.value))}
              placeholder="0.5"
            />
          </Field>
        </div>
      );

    case "binomial":
      return (
        <div className="grid grid-cols-2 gap-4">
          <Field label="Trials (n)" hint="Positive integer">
            <input
              className="input-field"
              type="number"
              min={1}
              value={num("n") ?? ""}
              onChange={(e) => set("n", parseInt(e.target.value))}
              placeholder="10"
            />
          </Field>
          <Field label="Probability (p)" hint="0.0 to 1.0">
            <input
              className="input-field"
              type="number"
              step="0.01"
              min={0}
              max={1}
              value={num("p") ?? ""}
              onChange={(e) => set("p", parseFloat(e.target.value))}
              placeholder="0.5"
            />
          </Field>
        </div>
      );

    case "poisson":
      return (
        <div className="max-w-xs">
          <Field label="Mean (lambda)" hint="Rate parameter, must be positive">
            <input
              className="input-field"
              type="number"
              step="any"
              min={0}
              value={num("mean") ?? ""}
              onChange={(e) => set("mean", parseFloat(e.target.value))}
              placeholder="5.0"
            />
          </Field>
        </div>
      );

    case "bernoulli_mixture":
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Field label="Probability (p)" hint="0.0 to 1.0">
              <input
                className="input-field"
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={num("p") ?? ""}
                onChange={(e) => set("p", parseFloat(e.target.value))}
                placeholder="0.5"
              />
            </Field>
            <Field label="Distribution Name" hint="scipy.stats distribution">
              <input
                className="input-field"
                value={str("dist_name")}
                onChange={(e) => set("dist_name", e.target.value)}
                placeholder="e.g. norm, gamma, expon"
              />
            </Field>
          </div>
          <Field label="Distribution Params (JSON)" hint='e.g. {"loc": 0, "scale": 1}'>
            <input
              className="input-field font-mono"
              value={
                typeof params.dist_params === "object"
                  ? JSON.stringify(params.dist_params)
                  : str("dist_params")
              }
              onChange={(e) => {
                try {
                  set("dist_params", JSON.parse(e.target.value));
                } catch {
                  set("dist_params", e.target.value);
                }
              }}
              placeholder='{"loc": 0, "scale": 1}'
            />
          </Field>
        </div>
      );

    case "scipy":
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Field label="Distribution Name" hint="scipy.stats distribution">
              <input
                className="input-field"
                value={str("dist_name")}
                onChange={(e) => set("dist_name", e.target.value)}
                placeholder="e.g. beta, gamma, lognorm"
              />
            </Field>
            <Field label="Decimal Places" hint="Optional">
              <input
                className="input-field"
                type="number"
                min={0}
                value={num("decimal_places") ?? ""}
                onChange={(e) =>
                  set("decimal_places", e.target.value ? parseInt(e.target.value) : undefined)
                }
                placeholder="auto"
              />
            </Field>
          </div>
          <Field label="Distribution Params (JSON)" hint='e.g. {"a": 2, "b": 5}'>
            <input
              className="input-field font-mono"
              value={
                typeof params.dist_params === "object"
                  ? JSON.stringify(params.dist_params)
                  : str("dist_params")
              }
              onChange={(e) => {
                try {
                  set("dist_params", JSON.parse(e.target.value));
                } catch {
                  set("dist_params", e.target.value);
                }
              }}
              placeholder='{"a": 2, "b": 5}'
            />
          </Field>
        </div>
      );

    case "person":
      return (
        <div className="grid grid-cols-3 gap-4">
          <Field label="Locale">
            <input
              className="input-field"
              value={str("locale", "en_US")}
              onChange={(e) => set("locale", e.target.value)}
              placeholder="en_US"
            />
          </Field>
          <Field label="Sex" hint="Optional filter">
            <select
              className="select-field"
              value={str("sex")}
              onChange={(e) => set("sex", e.target.value || undefined)}
            >
              <option value="">Any</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </Field>
          <Field label="With Personas">
            <label className="flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                checked={bool("with_synthetic_personas")}
                onChange={(e) => set("with_synthetic_personas", e.target.checked)}
                className="accent-[#76b900]"
              />
              <span className="text-sm text-gray-300">Include persona data</span>
            </label>
          </Field>
        </div>
      );

    case "person_from_faker":
      return (
        <div className="grid grid-cols-2 gap-4">
          <Field label="Locale">
            <input
              className="input-field"
              value={str("locale", "en_US")}
              onChange={(e) => set("locale", e.target.value)}
              placeholder="en_US"
            />
          </Field>
          <Field label="Sex" hint="Optional filter">
            <select
              className="select-field"
              value={str("sex")}
              onChange={(e) => set("sex", e.target.value || undefined)}
            >
              <option value="">Any</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </Field>
        </div>
      );

    default:
      return (
        <p className="text-sm text-gray-500">
          No parameter form available for sampler type "{samplerType}".
          Parameters will be sent as-is.
        </p>
      );
  }
}
