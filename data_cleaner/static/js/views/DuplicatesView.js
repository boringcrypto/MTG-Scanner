import { ref, computed, onMounted, watch } from "vue";

const PAGE_SIZE = 20;

const RULE_LABELS = {
  token_same_set_number: "Token – same set & number (double-faced)",
};

export default {
  template: `
    <div>
      <div class="d-flex align-items-center mb-3 gap-2">
        <router-link to="/" class="btn btn-outline-secondary btn-sm">
          <i class="bi bi-arrow-left"></i> Back
        </router-link>
        <h5 class="mb-0"><i class="bi bi-copy me-2"></i>Duplicate Groups</h5>
      </div>

      <div v-if="loading" class="text-center py-5">
        <div class="spinner-border text-secondary"></div>
        <p class="mt-2 text-muted">Loading…</p>
      </div>

      <div v-else-if="error" class="alert alert-danger">{{ error }}</div>

      <template v-else>
        <!-- Stats bar -->
        <div class="d-flex flex-wrap gap-3 mb-3 align-items-center">
          <span class="badge text-bg-secondary fs-6">
            {{ stats.total_groups }} total groups
          </span>
          <span class="badge text-bg-success fs-6">
            {{ stats.resolved }} resolved
          </span>
          <span class="badge text-bg-warning text-dark fs-6">
            {{ stats.unresolved }} unresolved
          </span>
          <div class="form-check form-switch ms-auto mb-0">
            <input class="form-check-input" type="checkbox" id="showResolved"
                   v-model="showResolved" />
            <label class="form-check-label" for="showResolved">Show resolved</label>
          </div>
        </div>

        <p class="text-muted small mb-3">
          Showing {{ activeGroups.length }} groups
          (page {{ page + 1 }} / {{ totalPages || 1 }}):
          {{ pageStart + 1 }}–{{ Math.min(pageStart + PAGE_SIZE, activeGroups.length) }}
        </p>

        <!-- Group cards -->
        <div v-for="(group, gi) in pageGroups" :key="gi"
             class="card mb-3 shadow-sm"
             :class="group.rule ? 'border-success' : ''">
          <div class="card-header py-1 small d-flex justify-content-between align-items-center"
               :class="group.rule ? 'bg-success bg-opacity-10' : ''">
            <span>
              Group {{ pageStart + gi + 1 }}
              &mdash; {{ group.indices.length }} identical cards
            </span>
            <span v-if="group.rule" class="badge text-bg-success">
              {{ ruleLabel(group.rule) }}
            </span>
            <span v-else class="badge text-bg-warning text-dark">unresolved</span>
          </div>
          <div class="card-body py-2 d-flex flex-wrap gap-2">
            <router-link v-for="idx in group.indices" :key="idx"
                         :to="'/card/' + idx"
                         class="text-decoration-none">
              <div class="text-center" style="width:80px">
                <img :src="'/api/card/' + idx + '/image'"
                     style="width:80px;height:112px;object-fit:cover;border-radius:4px"
                     :alt="'Card ' + idx" loading="lazy" />
                <div class="text-muted" style="font-size:10px">#{{ idx }}</div>
              </div>
            </router-link>
          </div>
        </div>

        <p v-if="activeGroups.length === 0" class="text-muted text-center py-5">
          No groups to show.
        </p>

        <!-- Pagination -->
        <nav v-if="totalPages > 1">
          <ul class="pagination justify-content-center flex-wrap">
            <li class="page-item" :class="{ disabled: page === 0 }">
              <button class="page-link" @click="page--">‹</button>
            </li>
            <li v-for="p in totalPages" :key="p"
                class="page-item" :class="{ active: page === p - 1 }">
              <button class="page-link" @click="page = p - 1">{{ p }}</button>
            </li>
            <li class="page-item" :class="{ disabled: page === totalPages - 1 }">
              <button class="page-link" @click="page++">›</button>
            </li>
          </ul>
        </nav>
      </template>
    </div>
  `,
  setup() {
    const allGroups   = ref([]);   // unified list: { indices, rule }
    const stats       = ref({ total_groups: 0, resolved: 0, unresolved: 0 });
    const loading     = ref(true);
    const error       = ref(null);
    const page        = ref(0);
    const showResolved = ref(false);

    const activeGroups = computed(() =>
      showResolved.value
        ? allGroups.value
        : allGroups.value.filter(g => !g.rule)
    );
    const totalPages = computed(() => Math.ceil(activeGroups.value.length / PAGE_SIZE));
    const pageStart  = computed(() => page.value * PAGE_SIZE);
    const pageGroups = computed(() =>
      activeGroups.value.slice(pageStart.value, pageStart.value + PAGE_SIZE)
    );

    function ruleLabel(rule) {
      return RULE_LABELS[rule] ?? rule;
    }

    onMounted(async () => {
      try {
        const r = await fetch("/api/duplicates");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        stats.value = d.stats;
        // Normalise both resolved and unresolved into the same shape
        const resolved   = (d.resolved   || []).map(g => ({ indices: g.indices,   rule: g.rule }));
        const unresolved = (d.unresolved || []).map(g => ({ indices: g,            rule: null  }));
        allGroups.value  = [...unresolved, ...resolved];
      } catch (e) {
        error.value = `Failed to load duplicates: ${e.message}`;
      } finally {
        loading.value = false;
      }
    });

    watch(showResolved, () => { page.value = 0; });

    return {
      allGroups, stats, loading, error, page, showResolved,
      PAGE_SIZE, activeGroups, totalPages, pageStart, pageGroups, ruleLabel,
    };
  },
};

