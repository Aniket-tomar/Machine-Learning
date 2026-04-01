# Design System Specification: High-End Technical Editorial

## 1. Overview & Creative North Star: "The Clinical Lens"
This design system moves beyond the "SaaS-template" aesthetic to create a high-end, editorial experience tailored for machine learning. The Creative North Star is **"The Clinical Lens"**—a philosophy that treats data as fine art and the interface as a precision instrument.

By rejecting traditional structural lines and rigid grids, we create a "liquid" technical environment. We achieve sophistication through **intentional asymmetry**, where large typographic headers anchor expansive white space, and **tonal layering**, where depth is felt rather than seen. This system is designed to provide technical clarity for complex ML models while maintaining the premium feel of a high-end digital publication.

---

## 2. Color & Atmospheric Depth
Our palette is a sophisticated blend of deep nautical blues and technical teals, anchored by a "cool-smoke" neutral scale.

### The "No-Line" Rule
**Standard 1px solid borders are strictly prohibited for sectioning.** To define boundaries, designers must use background shifts. For instance, a main content area using `surface` (`#f9f9ff`) should be delineated from a sidebar using `surface-container-low` (`#f0f3ff`). Physicality is created through color transitions, not ink lines.

### Surface Hierarchy & Nesting
Treat the UI as a series of stacked, semi-transparent layers. Use the surface tiers to create "nested" importance:
- **Base Layer:** `background` (`#f9f9ff`)
- **Secondary Sectioning:** `surface-container-low` (`#f0f3ff`)
- **Interactive Cards:** `surface-container-lowest` (`#ffffff`) to create a "pop" against the tinted background.
- **Overlays/Modals:** `surface-container-high` (`#dee8ff`) or `highest` (`#d8e3fb`).

### The "Glass & Gradient" Rule
To add "soul" to the technical aesthetic, use **Glassmorphism** for floating utility panels. Apply `surface-container-lowest` at 70% opacity with a `backdrop-blur` of 20px. 
**Signature Gradients:** For primary CTAs and data highlights, use a linear gradient from `primary` (`#004cca`) to `primary-container` (`#0062ff`) at a 135° angle. This adds a subtle 3D luminosity that flat colors cannot replicate.

---

## 3. Typography: Technical Authority
We use **Inter** for its mathematical precision and neutral "voice."

*   **Display Scale (`display-lg` to `display-sm`):** Reserved for high-level data insights or hero sections. Use `on-surface` (`#111c2d`) with a `-0.02em` letter spacing to feel "tighter" and more authoritative.
*   **Headline & Title (`headline-lg` to `title-sm`):** These serve as the structural anchors. In an editorial layout, use `headline-lg` (2rem) asymmetrically—pushed to the far left or right—to break the flow of standard data tables.
*   **Body & Label (`body-lg` to `label-sm`):** `body-md` (0.875rem) is our workhorse for technical descriptions. Use `on-surface-variant` (`#424656`) for secondary text to maintain a low-contrast, easy-reading experience.

---

## 4. Elevation & Depth: Tonal Layering
In "The Clinical Lens," we avoid the "heavy lifting" of dark shadows.

*   **The Layering Principle:** Place a `surface-container-lowest` card on a `surface-container-low` section. This creates a "soft lift" that feels architectural rather than digital.
*   **Ambient Shadows:** If an element must float (e.g., a dropdown), use an ultra-diffused shadow: `box-shadow: 0 10px 40px rgba(17, 28, 45, 0.06)`. Note the use of the `on-surface` color (#111c2d) at a very low opacity to mimic natural light.
*   **The "Ghost Border" Fallback:** If a container holds complex data (like a scatter plot) and needs a border for clarity, use the `outline-variant` (`#c2c6d9`) at **15% opacity**. It should be barely perceptible.

---

## 5. Components

### Buttons
*   **Primary:** Gradient of `primary` to `primary-container`. `rounded-md` (0.375rem). No border.
*   **Secondary:** `surface-container-highest` background with `on-primary-fixed-variant` text.
*   **Tertiary:** Ghost style. No background; text in `primary`. Underline only on hover.

### Data Visualization Cards
**Forbid the use of divider lines.** Separate metrics using vertical white space (Spacing `8` or `10`). Use `surface-container-lowest` as the card base. If multiple cards are grouped, use a 1px "Ghost Border" to define the group, not the individual cards.

### Input Fields
*   **States:** Default state uses `surface-container-high` background. On focus, transition to `surface-container-lowest` with a 1px `primary` "Ghost Border" (20% opacity).
*   **Typography:** Labels must use `label-md` in `on-surface-variant`.

### Technical Chips
*   **ML Status Chips:** Use `tertiary-container` for "Success/Active" with `on-tertiary-container` text. Use `rounded-full` for a "pill" look that contrasts with the architectural squareness of the rest of the UI.

### Additional: The "Inference Stream" List
For real-time ML logs, use a list where items are separated by a background shift from `surface` to `surface-container-low` on hover, rather than lines. This maintains a "clean stream" of data.

---

## 6. Do's and Don'ts

### Do:
*   **Embrace Negative Space:** Use spacing `16` (4rem) and `20` (5rem) to separate major technical modules.
*   **Use Asymmetry:** Place a large metric (`display-md`) off-center to create visual interest.
*   **Layer Surface Tones:** Use the full range of `surface-container` tokens to create a "landscape" for your data.

### Don't:
*   **Don't use 100% Black:** Never use `#000000`. Use `on-surface` (`#111c2d`) for all "black" text to maintain the cool, nautical depth.
*   **Don't use Hard Borders:** If you see a solid, high-contrast line, delete it. Use a `surface` shift instead.
*   **Don't Over-Shadow:** If a shadow looks "fuzzy" or "dirty," the opacity is too high. Keep it under 8%.