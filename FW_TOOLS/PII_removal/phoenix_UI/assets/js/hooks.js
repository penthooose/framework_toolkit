let Hooks = {};

Hooks.RegexInput = {
  mounted() {
    // Hook is mounted
  },
};

Hooks.DropdownPosition = {
  mounted() {
    this.updateDropdowns();
  },
  updated() {
    this.updateDropdowns();
  },
  updateDropdowns() {
    const dropdowns = document.querySelectorAll("[data-dropdown]");
    dropdowns.forEach((dropdown) => {
      const trigger = document.getElementById(dropdown.dataset.trigger);
      if (trigger) {
        const rect = trigger.getBoundingClientRect();

        // Ensure proper encoding for text content
        dropdown.querySelectorAll(".text-sm").forEach((element) => {
          element.textContent = decodeURIComponent(
            encodeURIComponent(element.textContent)
          );
        });

        const containerRect = this.el.getBoundingClientRect();

        dropdown.style.position = "fixed";
        dropdown.style.top = `${rect.bottom + window.scrollY}px`;
        dropdown.style.left = `${rect.left + window.scrollX}px`;
        dropdown.style.width = "2000px"; // Fixed width independent of trigger
        dropdown.style.zIndex = "50";
      }
    });
  },
};

Hooks.FileInput = {
  mounted() {
    this.el.addEventListener("change", (e) => {
      const target = this.el.dataset.target;
      const path = e.target.files[0]?.path;
      if (path) {
        this.pushEvent("set_" + target, { path: path });
      }
    });
  },
};

export default Hooks;
