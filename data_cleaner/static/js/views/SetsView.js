import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";

export default {
  template: `
    <div>
      <div class="d-flex align-items-center mb-3 gap-2 flex-wrap">
        <h4 class="mb-0 me-auto"><i class="bi bi-collection me-2"></i>Browse by Set</h4>
        <input v-model="search" type="search" class="form-control w-auto"
               placeholder="Search sets…" style="min-width:200px" />
        <select v-model="sortBy" class="form-select w-auto">
          <option value="name">Sort: Name</option>
          <option value="release_date">Sort: Release date</option>
          <option value="canonical_card_count">Sort: Card count</option>
        </select>
      </div>

      <div v-if="loading" class="text-center py-5">
        <div class="spinner-border text-secondary" role="status"></div>
      </div>

      <div v-else-if="error" class="alert alert-danger">{{ error }}</div>

      <div v-else>
        <p class="text-muted small mb-2">
          Showing {{ filtered.length.toLocaleString() }} of
          {{ sets.length.toLocaleString() }} sets
        </p>

        <div class="table-responsive">
          <table class="table table-sm table-hover align-middle">
            <thead class="table-dark sticky-top">
              <tr>
                <th>Set name</th>
                <th>Code</th>
                <th>Series</th>
                <th>Release</th>
                <th class="text-end">Total cards</th>
                <th class="text-end">Canonical</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="s in filtered" :key="s.set_id"
                  class="cursor-pointer"
                  @click="go(s.set_id)">
                <td>{{ s.name }}</td>
                <td><span class="badge bg-secondary font-monospace">{{ s.set_code }}</span></td>
                <td class="text-muted small">{{ s.series || '—' }}</td>
                <td class="text-muted small">{{ s.release_date || '—' }}</td>
                <td class="text-end">{{ (s.total_cards || 0).toLocaleString() }}</td>
                <td class="text-end fw-semibold">{{ s.canonical_card_count.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `,
  setup() {
    const router  = useRouter();
    const sets    = ref([]);
    const loading = ref(true);
    const error   = ref(null);
    const search  = ref("");
    const sortBy  = ref("name");

    onMounted(async () => {
      try {
        const r = await fetch("/api/sets");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        sets.value = await r.json();
      } catch (e) {
        error.value = e.message;
      } finally {
        loading.value = false;
      }
    });

    const filtered = computed(() => {
      const q = search.value.trim().toLowerCase();
      let list = q
        ? sets.value.filter(s =>
            (s.name || "").toLowerCase().includes(q) ||
            (s.set_code || "").toLowerCase().includes(q) ||
            (s.series || "").toLowerCase().includes(q))
        : sets.value.slice();

      return list.sort((a, b) => {
        if (sortBy.value === "release_date")
          return (b.release_date || "").localeCompare(a.release_date || "");
        if (sortBy.value === "canonical_card_count")
          return b.canonical_card_count - a.canonical_card_count;
        return (a.name || "").localeCompare(b.name || "");
      });
    });

    function go(set_id) {
      router.push(`/set/${encodeURIComponent(set_id)}`);
    }

    return { sets, loading, error, search, sortBy, filtered, go };
  },
};
