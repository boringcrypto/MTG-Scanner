import { ref, computed, onMounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { cardCount, fetchCount } from "../store.js";

export default {
  template: `
    <div>
      <div class="d-flex align-items-center mb-3 gap-2">
        <button class="btn btn-outline-secondary btn-sm" @click="$router.back()">
          <i class="bi bi-arrow-left"></i> Back
        </button>
        <h5 class="mb-0">
          Card <strong>#{{ idx }}</strong>
          <span v-if="info" class="text-muted fw-normal fs-6 ms-2">{{ info.name }}</span>
        </h5>
      </div>

      <div class="row g-4">
        <!-- Image -->
        <div class="col-auto">
          <div class="card shadow-sm" style="width:fit-content">
            <div v-if="imgLoading" class="d-flex align-items-center justify-content-center"
                 style="width:223px;height:310px">
              <div class="spinner-border text-secondary"></div>
            </div>
            <img v-else-if="!imgError" :src="imgSrc" class="card-img-top rounded"
                 style="width:223px" :alt="'Card ' + idx" />
            <div v-else class="card-body text-danger p-3">
              <i class="bi bi-exclamation-triangle me-1"></i>{{ imgError }}
            </div>
          </div>
          <div class="d-flex gap-2 mt-2">
            <button class="btn btn-outline-primary btn-sm flex-fill"
                    :disabled="idx <= 0" @click="go(idx - 1)">
              <i class="bi bi-chevron-left"></i> Prev
            </button>
            <button class="btn btn-outline-primary btn-sm flex-fill"
                    :disabled="cardCount !== null && idx >= cardCount - 1"
                    @click="go(idx + 1)">
              Next <i class="bi bi-chevron-right"></i>
            </button>
          </div>
        </div>

        <!-- Info -->
        <div class="col">
          <div v-if="infoLoading" class="text-muted">
            <div class="spinner-border spinner-border-sm me-2"></div>Loading info…
          </div>
          <div v-else-if="infoError" class="alert alert-warning py-2">{{ infoError }}</div>
          <template v-else-if="info">
            <div class="mb-2 d-flex flex-wrap gap-1">
              <span v-if="info.rarity"    class="badge text-bg-secondary">{{ info.rarity }}</span>
              <span v-if="info.set_code"  class="badge text-bg-light border">{{ info.set_code }}</span>
              <span v-if="info.color"     class="badge text-bg-success">{{ info.color }}</span>
              <span v-if="info.duplicate" class="badge text-bg-warning">duplicate</span>
              <span v-if="!info.found"    class="badge text-bg-danger">not in DB</span>
            </div>
            <dl class="row g-0 small mb-0">
              <template v-for="[label, val] in fields" :key="label">
                <dt class="col-4 text-muted fw-normal text-truncate pe-2">{{ label }}</dt>
                <dd class="col-8 mb-1">{{ val }}</dd>
              </template>
            </dl>
            <div v-if="info.description" class="mt-3">
              <p class="text-muted small mb-1">Rules text</p>
              <p class="small" style="white-space:pre-wrap">{{ info.description }}</p>
            </div>
            <div v-if="info.flavor_text" class="mt-1">
              <p class="text-muted small mb-1 fst-italic">Flavour</p>
              <p class="small fst-italic" style="white-space:pre-wrap">{{ info.flavor_text }}</p>
            </div>
          </template>
          <div v-else class="text-muted small">No database record for this index.</div>
        </div>
      </div>
    </div>
  `,
  setup() {
    const route  = useRoute();
    const router = useRouter();

    const imgLoading  = ref(true);
    const imgError    = ref(null);
    const infoLoading = ref(false);
    const infoError   = ref(null);
    const info        = ref(null);

    const idx    = computed(() => parseInt(route.params.idx, 10));
    const imgSrc = computed(() => `/api/card/${idx.value}/image`);

    const fields = computed(() => {
      if (!info.value) return [];
      const d = info.value;
      return [
        ["Product ID",  d.product_id],
        ["Set",         d.set_name],
        ["Set code",    d.set_code],
        ["Number",      d.number],
        ["Type",        d.full_type || d.card_type || d.type],
        ["CMC",         d.converted_cost],
        ["Power/Tough", (d.power != null && d.toughness != null) ? `${d.power}/${d.toughness}` : null],
        ["Market $",    d.market_price != null ? `$${d.market_price.toFixed(2)}` : null],
        ["Mid $",       d.mid_price    != null ? `$${d.mid_price.toFixed(2)}`    : null],
        ["Low $",       d.low_price    != null ? `$${d.low_price.toFixed(2)}`    : null],
      ].filter(([, v]) => v != null);
    });

    function loadImage() {
      imgLoading.value = true;
      imgError.value   = null;
      const img = new Image();
      img.onload  = () => { imgLoading.value = false; };
      img.onerror = () => { imgLoading.value = false; imgError.value = "Failed to load image"; };
      img.src = imgSrc.value;
    }

    async function loadInfo() {
      infoLoading.value = true;
      infoError.value   = null;
      info.value        = null;
      try {
        const r = await fetch(`/api/card/${idx.value}/info`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        info.value = await r.json();
      } catch (e) {
        infoError.value = `Could not load card info: ${e.message}`;
      } finally {
        infoLoading.value = false;
      }
    }

    function go(n) { router.push(`/card/${n}`); }

    onMounted(() => { fetchCount(); loadImage(); loadInfo(); });
    watch(idx, () => { loadImage(); loadInfo(); });

    return { idx, imgSrc, imgLoading, imgError, infoLoading, infoError, info, fields, cardCount, go };
  },
};
