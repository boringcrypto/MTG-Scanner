import { ref, watch, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";

function useCard(idxRef) {
  const info       = ref(null);
  const infoError  = ref("");
  const imgLoading = ref(true);
  const imgError   = ref("");
  const imgSrc     = ref("");

  async function load(idx) {
    info.value      = null;
    infoError.value = "";
    imgLoading.value = true;
    imgError.value  = "";
    imgSrc.value    = "";

    if (idx === null || idx === "" || isNaN(idx)) return;

    // Info
    try {
      const res = await fetch(`/api/card/${idx}/info`);
      if (!res.ok) { infoError.value = `HTTP ${res.status}`; }
      else info.value = await res.json();
    } catch (e) { infoError.value = "Failed to load info."; }

    // Image
    const img = new Image();
    img.onload  = () => { imgLoading.value = false; };
    img.onerror = () => { imgLoading.value = false; imgError.value = "Image not found."; };
    img.src     = `/api/card/${idx}/image`;
    imgSrc.value = img.src;
  }

  watch(idxRef, (v) => load(v), { immediate: true });

  return { info, infoError, imgLoading, imgError, imgSrc };
}

const INFO_FIELDS = [
  ["Product ID", "product_id"], ["Name", "name"], ["Set", "set_name"],
  ["Number", "number"], ["Rarity", "rarity"], ["Type", "card_type"],
  ["Color", "color"], ["Cost", "converted_cost"],
  ["Power / Toughness", (d) => d.power != null ? `${d.power} / ${d.toughness}` : null],
  ["Market price", (d) => d.market_price != null ? `$${d.market_price}` : null],
];

export default {
  template: `
    <div>
      <div class="d-flex align-items-center mb-3 gap-2">
        <router-link to="/" class="btn btn-outline-secondary btn-sm">
          <i class="bi bi-arrow-left"></i> Back
        </router-link>
        <h5 class="mb-0"><i class="bi bi-layout-split me-2"></i>Compare Cards</h5>
      </div>

      <!-- Index entry form -->
      <div class="card shadow-sm mb-4">
        <div class="card-body">
          <form class="row g-2 align-items-end" @submit.prevent="applyInputs">
            <div class="col-auto">
              <label class="form-label mb-1 small">Index A</label>
              <input v-model.number="inputA" type="number" min="0" class="form-control" style="width:120px" />
            </div>
            <div class="col-auto">
              <label class="form-label mb-1 small">Index B</label>
              <input v-model.number="inputB" type="number" min="0" class="form-control" style="width:120px" />
            </div>
            <div class="col-auto">
              <button class="btn btn-primary" type="submit">Compare</button>
            </div>
          </form>
        </div>
      </div>

      <!-- Success / error banner -->
      <div v-if="success" class="alert alert-success d-flex justify-content-between align-items-center">
        <span><i class="bi bi-check-circle me-2"></i>{{ success }}</span>
        <button class="btn-close" @click="success = ''"></button>
      </div>
      <div v-if="markError" class="alert alert-danger d-flex justify-content-between align-items-center">
        <span><i class="bi bi-exclamation-triangle me-2"></i>{{ markError }}</span>
        <button class="btn-close" @click="markError = ''"></button>
      </div>

      <!-- Side-by-side -->
      <div class="row g-4" v-if="idxA !== null && idxB !== null">

        <!-- Card A -->
        <div class="col-md-6">
          <div class="card h-100 shadow-sm">
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
              <strong>#{{ idxA }}</strong>
              <button class="btn btn-success btn-sm" :disabled="marking"
                      @click="markCanonical(idxA, idxB)">
                <i class="bi bi-star-fill me-1"></i>Mark as Canonical
              </button>
            </div>
            <div class="card-body d-flex flex-column gap-3">
              <div class="text-center">
                <div v-if="a.imgLoading.value" class="d-flex justify-content-center align-items-center"
                     style="height:620px">
                  <div class="spinner-border text-secondary"></div>
                </div>
                <img v-else-if="!a.imgError.value"
                     :src="peeking && b.imgSrc.value ? b.imgSrc.value : a.imgSrc.value"
                     @mousedown.prevent="peeking = true"
                     @mouseup="peeking = false"
                     @mouseleave="peeking = false"
                     style="max-width:446px;border-radius:6px;cursor:pointer;user-select:none" />
                <div v-else class="text-danger small">{{ a.imgError.value }}</div>
                <div class="text-muted" style="font-size:0.7rem;margin-top:4px">Hold to peek at other card</div>
              </div>
              <dl class="row g-0 small mb-0" v-if="a.info.value">
                <template v-for="[label, key] in fields" :key="label">
                  <dt class="col-5 text-muted fw-normal text-truncate pe-2">{{ label }}</dt>
                  <dd class="col-7 mb-1">{{ fieldVal(a.info.value, key) }}</dd>
                </template>
              </dl>
              <div v-else-if="a.infoError.value" class="text-danger small">{{ a.infoError.value }}</div>
              <div v-else class="text-muted small">Loading…</div>
            </div>
          </div>
        </div>

        <!-- Card B -->
        <div class="col-md-6">
          <div class="card h-100 shadow-sm">
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
              <strong>#{{ idxB }}</strong>
              <button class="btn btn-success btn-sm" :disabled="marking"
                      @click="markCanonical(idxB, idxA)">
                <i class="bi bi-star-fill me-1"></i>Mark as Canonical
              </button>
            </div>
            <div class="card-body d-flex flex-column gap-3">
              <div class="text-center">
                <div v-if="b.imgLoading.value" class="d-flex justify-content-center align-items-center"
                     style="height:620px">
                  <div class="spinner-border text-secondary"></div>
                </div>
                <img v-else-if="!b.imgError.value"
                     :src="peeking && a.imgSrc.value ? a.imgSrc.value : b.imgSrc.value"
                     @mousedown.prevent="peeking = true"
                     @mouseup="peeking = false"
                     @mouseleave="peeking = false"
                     style="max-width:446px;border-radius:6px;cursor:pointer;user-select:none" />
                <div v-else class="text-danger small">{{ b.imgError.value }}</div>
                <div class="text-muted" style="font-size:0.7rem;margin-top:4px">Hold to peek at other card</div>
              </div>
              <dl class="row g-0 small mb-0" v-if="b.info.value">
                <template v-for="[label, key] in fields" :key="label">
                  <dt class="col-5 text-muted fw-normal text-truncate pe-2">{{ label }}</dt>
                  <dd class="col-7 mb-1">{{ fieldVal(b.info.value, key) }}</dd>
                </template>
              </dl>
              <div v-else-if="b.infoError.value" class="text-danger small">{{ b.infoError.value }}</div>
              <div v-else class="text-muted small">Loading…</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  `,

  setup() {
    const route  = useRoute();
    const router = useRouter();

    const idxA   = ref(null);
    const idxB   = ref(null);
    const inputA = ref("");
    const inputB = ref("");
    const marking   = ref(false);
    const success   = ref("");
    const markError = ref("");
    const peeking   = ref(false);

    function applyInputs() {
      if (inputA.value === "" || inputB.value === "") return;
      router.replace({ path: "/compare", query: { a: inputA.value, b: inputB.value } });
    }

    onMounted(() => {
      const qa = parseInt(route.query.a);
      const qb = parseInt(route.query.b);
      if (!isNaN(qa)) { idxA.value = qa; inputA.value = qa; }
      if (!isNaN(qb)) { idxB.value = qb; inputB.value = qb; }
    });

    // Update reactive indices when route query changes
    watch(() => route.query, (q) => {
      const qa = parseInt(q.a);
      const qb = parseInt(q.b);
      idxA.value = isNaN(qa) ? null : qa;
      idxB.value = isNaN(qb) ? null : qb;
      if (!isNaN(qa)) inputA.value = qa;
      if (!isNaN(qb)) inputB.value = qb;
    });

    const a = useCard(idxA);
    const b = useCard(idxB);

    const fields = INFO_FIELDS;

    function fieldVal(info, key) {
      if (typeof key === "function") return key(info) ?? "—";
      return info[key] ?? "—";
    }

    async function markCanonical(canonicalIdx, duplicateIdx) {
      success.value   = "";
      markError.value = "";
      marking.value   = true;
      try {
        const res = await fetch("/api/compare/mark-duplicate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ canonical_idx: canonicalIdx, duplicate_idx: duplicateIdx }),
        });
        const data = await res.json();
        if (!res.ok) { markError.value = data.error || `HTTP ${res.status}`; }
        else success.value = `#${duplicateIdx} marked as duplicate of #${canonicalIdx}.`;
      } catch (e) {
        markError.value = "Request failed.";
      } finally {
        marking.value = false;
      }
    }

    return { idxA, idxB, inputA, inputB, a, b, fields, fieldVal,
             marking, success, markError, markCanonical, applyInputs, peeking };
  },
};
