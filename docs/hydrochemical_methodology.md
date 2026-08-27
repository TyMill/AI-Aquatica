# Hydrochemical charge-balance methodology

AI-Aquatica evaluates the electrical charge closure of a selected set of dissolved cations and anions. The procedure is intended as a transparent quality-control diagnostic; it does not replace laboratory validation or justify automatic correction of analytical measurements.

## Units and conversion

Charge-balance calculations are performed in milliequivalents per litre (meq/L). When an ion concentration is supplied in mg/L, AI-Aquatica converts it using:

\[
c_{i,\mathrm{meq/L}} = \frac{c_{i,\mathrm{mg/L}}}{EW_i}
\]

where \(EW_i\) is the ion-specific equivalent weight in mg/meq:

\[
EW_i = \frac{M_i}{|z_i|}
\]

Here, \(M_i\) is the molar mass and \(|z_i|\) is the absolute ionic charge. Built-in equivalent weights are defined explicitly in `src/ai_aquatica/hydrochemistry/ion_balance.py` and can be overridden by the user.

Examples include:

| Ion | Equivalent weight (mg/meq) |
|---|---:|
| Ca | 20.039 |
| Mg | 12.1525 |
| Na | 22.9898 |
| K | 39.0983 |
| NH4 | 18.039 |
| Cl | 35.453 |
| SO4 | 48.03 |
| NO3 | 62.0049 |
| NO2 | 46.0055 |
| HCO3 | 61.0168 |


### Constituents reported on an elemental basis

Some laboratories report nitrogen species as mass of nitrogen rather than mass of the complete ion. For monovalent `NO3-N`, `NO2-N`, and `NH4-N`, the conversion is:

\[
c_{\mathrm{meq/L}} = \frac{c_{\mathrm{mg\ N/L}}}{14.0067}
\]

because one milliequivalent of each species contains one millimole of nitrogen. AI-Aquatica keeps the default ion-based factors for general use and provides `NITROGEN_AS_N_MASS_PER_MEQ` for datasets reported as mg N/L. The SoftwareX dataset uses this explicit mapping.

## Alkalinity-derived bicarbonate

When bicarbonate is not measured directly, AI-Aquatica can derive HCO3 from alkalinity reported as mg CaCO3/L:

\[
\mathrm{HCO_3\ (mg/L)} = \mathrm{Alkalinity\ (mg\ CaCO_3/L)} \times \frac{61.0168}{50.043}
\]

For alkalinity already expressed in meq/L:

\[
\mathrm{HCO_3\ (mg/L)} = \mathrm{Alkalinity\ (meq/L)} \times 61.0168
\]

The derived bicarbonate value is preserved in a separate column. Users must verify the alkalinity unit before interpreting the result.

## Charge-balance error

The charge-balance error is calculated as:

\[
CBE(\%) = 100 \times \frac{\sum C - \sum A}{\sum C + \sum A}
\]

where \(\sum C\) and \(\sum A\) are the sums of selected cations and anions in meq/L.

The default threshold is 5%, but the threshold is configurable. The real-dataset SoftwareX workflow uses 10% because the illustrative dataset lacks sodium and potassium and therefore does not contain a complete conventional major-ion set. This threshold should be interpreted as a diagnostic criterion for the available ion set, not as proof of complete hydrochemical closure.

## Row-level validation and status

Every selected ion value is checked for:

- column availability;
- numeric convertibility;
- finite values;
- non-negative concentrations;
- completeness of the selected ion set;
- positive total ionic charge.

Each row receives one of three statuses:

- `acceptable`: a complete and evaluable row with absolute CBE at or below the threshold;
- `review`: a complete and evaluable row with absolute CBE above the threshold;
- `indeterminate`: at least one selected ion is missing or invalid, or the total ionic charge is zero.

An indeterminate row receives `NaN` for CBE and a nullable value for `Potential_Error`. It is never assigned an artificial 0% charge-balance error.

## Connection to machine learning

Predictive workflows can use the hydrochemical status through four explicit policies:

- `quality_policy="warn"`: retain all rows and emit a warning;
- `quality_policy="filter"`: retain only accepted statuses;
- `quality_policy="raise"`: stop model fitting when non-accepted rows are present;
- `quality_policy="ignore"`: proceed without applying the hydrochemical status.

This mechanism makes hydrochemical quality control operational rather than merely descriptive.
