import { ref, computed, onMounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";

const PAGE_SIZE = 500;

export default {
  template: `
    <div>
      <!-- Back nav -->
      <nav class="mb-3">
        <router-link to="/sets" class="btn btn-sm btn-outline-secondary">
          <i class="bi bi-arrow-left me-1"></i>All Sets
        </router-link>
      </nav>

      <div v-if="loading" class="text-center py-5">
        <div class="spinner-border text-secondary" role="status"></div>
      </div>

      <div v-else-if="error" class="alert alert-danger">{{ error }}</div>

      <div v-else>
        <!-- Set info header -->
        <div class="card shadow-sm mb-4">
          <div class="card-body">
            <h4 class="card-title mb-1">{{ setInfo.name }}</h4>
            <div class="d-flex flex-wrap gap-2 mt-2">
              <span class="badge bg-secondary font-monospace fs-6">{{ setInfo.set_code }}</span>
              <span v-if="setInfo.series" class="badge bg-info text-dark">{{ setInfo.series }}</span>
              <span v-if="setInfo.release_date" class="badge bg-light text-dark border">
                <i class="bi bi-calendar me-1"></i>{{ setInfo.release_date }}
              </span>
            </div>
            <div class="row mt-3 g-3">
              <div class="col-auto">
                <div class="text-muted small">Set ID</div>
                <div class="fw-semibold font-monospace">{{ setInfo.set_id }}</div>
              </div>
              <div class="col-auto" v-if="setInfo.total_cards">
                <div class="text-muted small">Total cards in set</div>
                <div class="fw-semibold">{{ setInfo.total_cards.toLocaleString() }}</div>
              </div>
              <div class="col-auto">
                <div class="text-muted small">Cards</div>
                <div class="fw-semibold text-success">{{ mainCards.length.toLocaleString() }}</div>
              </div>
              <div class="col-auto" v-if="specialCards.length">
                <div class="text-muted small">Special</div>
                <div class="fw-semibold" style="color:#6f42c1">{{ specialCards.length.toLocaleString() }}</div>
              </div>
              <div class="col-auto" v-if="tokenCards.length">
                <div class="text-muted small">Tokens</div>
                <div class="fw-semibold text-warning">{{ tokenCards.length.toLocaleString() }}</div>
              </div>
            </div>
          </div>
        </div>

        <div v-if="allCards.length === 0" class="text-muted">No canonical cards found for this set.</div>

        <div v-else>
          <!-- ── Main cards ── -->
          <template v-if="mainCards.length">
            <h5 class="mb-2">
              Cards
              <span class="badge bg-secondary ms-1">{{ mainCards.length.toLocaleString() }}</span>
            </h5>
            <div class="d-flex align-items-center justify-content-between mb-2 flex-wrap gap-2">
              <span class="text-muted small">
                {{ mainPageStart + 1 }}–{{ mainPageEnd }} of {{ mainCards.length.toLocaleString() }}
              </span>
              <div class="btn-group btn-group-sm">
                <button class="btn btn-outline-secondary" :disabled="mainPage === 0" @click="mainPage--">
                  <i class="bi bi-chevron-left"></i>
                </button>
                <button class="btn btn-outline-secondary" :disabled="mainPageEnd >= mainCards.length" @click="mainPage++">
                  <i class="bi bi-chevron-right"></i>
                </button>
              </div>
            </div>
            <div class="d-flex flex-wrap gap-2 mb-4">
              <router-link v-for="c in mainPageCards" :key="c.lmdb_idx"
                :to="'/card/' + c.lmdb_idx" class="text-decoration-none"
                :title="[c.number, c.name, c.rarity].filter(Boolean).join(' · ')">
                <div class="border rounded overflow-hidden bg-dark position-relative" style="width:160px">
                  <img :src="'/api/card/' + c.lmdb_idx + '/image'" loading="lazy"
                       width="160" height="224" style="display:block;object-fit:cover" />
                  <div class="position-absolute bottom-0 start-0 end-0 text-white text-center"
                       style="font-size:10px;background:rgba(0,0,0,.6);padding:1px 2px;line-height:1.3">
                    {{ c.number || c.lmdb_idx }}
                  </div>
                </div>
              </router-link>
            </div>
            <div class="d-flex justify-content-center mb-4 gap-1 flex-wrap" v-if="mainTotalPages > 1">
              <button v-for="p in mainTotalPages" :key="p" class="btn btn-sm"
                      :class="mainPage === p - 1 ? 'btn-primary' : 'btn-outline-secondary'"
                      @click="mainPage = p - 1">{{ p }}</button>
            </div>
          </template>

          <!-- ── Special ── -->
          <template v-if="specialCards.length">
            <hr v-if="mainCards.length" />
            <h5 class="mb-2">
              Special
              <span class="badge bg-purple ms-1" style="background:#6f42c1">{{ specialCards.length.toLocaleString() }}</span>
            </h5>
            <div class="d-flex align-items-center justify-content-between mb-2 flex-wrap gap-2">
              <span class="text-muted small">
                {{ specialPageStart + 1 }}–{{ specialPageEnd }} of {{ specialCards.length.toLocaleString() }}
              </span>
              <div class="btn-group btn-group-sm">
                <button class="btn btn-outline-secondary" :disabled="specialPage === 0" @click="specialPage--">
                  <i class="bi bi-chevron-left"></i>
                </button>
                <button class="btn btn-outline-secondary" :disabled="specialPageEnd >= specialCards.length" @click="specialPage++">
                  <i class="bi bi-chevron-right"></i>
                </button>
              </div>
            </div>
            <div class="d-flex flex-wrap gap-2 mb-4">
              <router-link v-for="c in specialPageCards" :key="c.lmdb_idx"
                :to="'/card/' + c.lmdb_idx" class="text-decoration-none"
                :title="[c.number, c.name, c.rarity].filter(Boolean).join(' · ')">
                <div class="rounded overflow-hidden bg-dark position-relative" style="width:160px;border:1px solid #6f42c1">
                  <img :src="'/api/card/' + c.lmdb_idx + '/image'" loading="lazy"
                       width="160" height="224" style="display:block;object-fit:cover" />
                  <div class="position-absolute bottom-0 start-0 end-0 text-white text-center"
                       style="font-size:10px;background:rgba(0,0,0,.6);padding:1px 2px;line-height:1.3">
                    {{ c.number || c.lmdb_idx }}
                  </div>
                </div>
              </router-link>
            </div>
            <div class="d-flex justify-content-center mb-4 gap-1 flex-wrap" v-if="specialTotalPages > 1">
              <button v-for="p in specialTotalPages" :key="p" class="btn btn-sm"
                      :class="specialPage === p - 1 ? 'btn-secondary' : 'btn-outline-secondary'"
                      @click="specialPage = p - 1">{{ p }}</button>
            </div>
          </template>

          <!-- ── Tokens ── -->
          <template v-if="tokenCards.length">
            <hr v-if="mainCards.length || specialCards.length" />
            <h5 class="mb-2">
              Tokens
              <span class="badge bg-warning text-dark ms-1">{{ tokenCards.length.toLocaleString() }}</span>
            </h5>
            <div class="d-flex align-items-center justify-content-between mb-2 flex-wrap gap-2">
              <span class="text-muted small">
                {{ tokenPageStart + 1 }}–{{ tokenPageEnd }} of {{ tokenCards.length.toLocaleString() }}
              </span>
              <div class="btn-group btn-group-sm">
                <button class="btn btn-outline-secondary" :disabled="tokenPage === 0" @click="tokenPage--">
                  <i class="bi bi-chevron-left"></i>
                </button>
                <button class="btn btn-outline-secondary" :disabled="tokenPageEnd >= tokenCards.length" @click="tokenPage++">
                  <i class="bi bi-chevron-right"></i>
                </button>
              </div>
            </div>
            <div class="d-flex flex-wrap gap-2 mb-4">
              <router-link v-for="c in tokenPageCards" :key="c.lmdb_idx"
                :to="'/card/' + c.lmdb_idx" class="text-decoration-none"
                :title="[c.number, c.name, c.rarity].filter(Boolean).join(' · ')">
                <div class="border border-warning rounded overflow-hidden bg-dark position-relative" style="width:160px">
                  <img :src="'/api/card/' + c.lmdb_idx + '/image'" loading="lazy"
                       width="160" height="224" style="display:block;object-fit:cover" />
                  <div class="position-absolute bottom-0 start-0 end-0 text-white text-center"
                       style="font-size:10px;background:rgba(0,0,0,.6);padding:1px 2px;line-height:1.3">
                    {{ c.number || c.lmdb_idx }}
                  </div>
                </div>
              </router-link>
            </div>
            <div class="d-flex justify-content-center mb-4 gap-1 flex-wrap" v-if="tokenTotalPages > 1">
              <button v-for="p in tokenTotalPages" :key="p" class="btn btn-sm"
                      :class="tokenPage === p - 1 ? 'btn-warning' : 'btn-outline-secondary'"
                      @click="tokenPage = p - 1">{{ p }}</button>
            </div>
          </template>
        </div>
      </div>
    </div>
  `,
  setup() {
    const route   = useRoute();
    const setInfo = ref({});
    const allCards = ref([]);
    const loading = ref(true);
    const error   = ref(null);
    const mainPage    = ref(0);
    const specialPage = ref(0);
    const tokenPage   = ref(0);

    async function load(set_id) {
      loading.value    = true;
      error.value      = null;
      mainPage.value    = 0;
      specialPage.value = 0;
      tokenPage.value   = 0;
      try {
        const [infoRes, cardsRes] = await Promise.all([
          fetch(`/api/set/${encodeURIComponent(set_id)}`),
          fetch(`/api/set/${encodeURIComponent(set_id)}/cards`),
        ]);
        if (!infoRes.ok)  throw new Error(`Set info HTTP ${infoRes.status}`);
        if (!cardsRes.ok) throw new Error(`Cards HTTP ${cardsRes.status}`);
        setInfo.value  = await infoRes.json();
        allCards.value = await cardsRes.json();
      } catch (e) {
        error.value = e.message;
      } finally {
        loading.value = false;
      }
    }

    onMounted(() => load(decodeURIComponent(route.params.set_id)));
    watch(() => route.params.set_id, id => id && load(decodeURIComponent(id)));

    function isToken(c) {
      return (c.rarity    || "").toLowerCase() === "token" ||
             (c.card_type || "").toLowerCase().includes("token");
    }
    function isSpecial(c) {
      return !isToken(c) && (c.rarity || "").toLowerCase() === "special";
    }
    const mainCards    = computed(() => allCards.value.filter(c => !isToken(c) && !isSpecial(c)));
    const specialCards = computed(() => allCards.value.filter(c =>  isSpecial(c)));
    const tokenCards   = computed(() => allCards.value.filter(c =>  isToken(c)));

    const mainPageStart  = computed(() => mainPage.value * PAGE_SIZE);
    const mainPageEnd    = computed(() => Math.min(mainPageStart.value + PAGE_SIZE, mainCards.value.length));
    const mainPageCards  = computed(() => mainCards.value.slice(mainPageStart.value, mainPageEnd.value));
    const mainTotalPages = computed(() => Math.ceil(mainCards.value.length / PAGE_SIZE));

    const specialPageStart  = computed(() => specialPage.value * PAGE_SIZE);
    const specialPageEnd    = computed(() => Math.min(specialPageStart.value + PAGE_SIZE, specialCards.value.length));
    const specialPageCards  = computed(() => specialCards.value.slice(specialPageStart.value, specialPageEnd.value));
    const specialTotalPages = computed(() => Math.ceil(specialCards.value.length / PAGE_SIZE));

    const tokenPageStart  = computed(() => tokenPage.value * PAGE_SIZE);
    const tokenPageEnd    = computed(() => Math.min(tokenPageStart.value + PAGE_SIZE, tokenCards.value.length));
    const tokenPageCards  = computed(() => tokenCards.value.slice(tokenPageStart.value, tokenPageEnd.value));
    const tokenTotalPages = computed(() => Math.ceil(tokenCards.value.length / PAGE_SIZE));

    return {
      setInfo, allCards, loading, error,
      mainCards,    mainPage,    mainPageStart,    mainPageEnd,    mainPageCards,    mainTotalPages,
      specialCards, specialPage, specialPageStart, specialPageEnd, specialPageCards, specialTotalPages,
      tokenCards,   tokenPage,   tokenPageStart,   tokenPageEnd,   tokenPageCards,   tokenTotalPages,
    };
  },
};
