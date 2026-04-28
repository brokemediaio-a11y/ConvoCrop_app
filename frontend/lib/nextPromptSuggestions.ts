type SuggestionCategory =
  | 'visual_detail'
  | 'severity'
  | 'spread'
  | 'management'
  | 'prevention'
  | 'scouting'
  | 'regional'
  | 'variety'
  | 'nutrition'
  | 'water'
  | 'seed'
  | 'economics'
  | 'mistakes'
  | 'confirm_healthy'
  | 'cure'
  | 'general_care'

interface MessageLike {
  role: 'user' | 'assistant'
  content: string
}

const SUGGESTION_POOLS: Record<SuggestionCategory, string[]> = {
  visual_detail: [
    'Can you describe the damage pattern you see in more detail?',
    'What exactly does the affected area look like?',
    'Describe the shape and colour of the marks on this leaf.',
    'What visual features confirm your diagnosis?',
    'How can I tell this apart from other rice diseases just by looking?',
    'Walk me through what you see on the leaf surface.',
    'What makes these marks different from other diseases?',
    'Describe the lesion edges and centres visible in this image.',
    'If I zoom in on the marks, what key features should I notice?',
    'What diagnostic visual clues are present in this leaf?',
    'Are there any subtle details in this image that helped your diagnosis?',
    'How do the marks in this image compare to a textbook case?',
  ],
  severity: [
    'How severe is this infection?',
    'On a scale from mild to severe, where does this case fall?',
    'Should I be worried about this level of damage?',
    'Is this considered an early or advanced stage?',
    'How bad is the damage visible in this image?',
    'Is this mild enough to recover or is it already serious?',
    'What severity level would you assign to this leaf?',
    'Given what you see, how urgent is treatment?',
    'Is this infection level typical or unusually heavy?',
    'How does this severity compare to what triggers treatment?',
    'At this level of damage, what is the likely yield impact?',
    'Would you say this needs immediate action or monitoring?',
  ],
  spread: [
    'How does this disease spread?',
    'What causes this disease to move from plant to plant?',
    'Can this infection spread to the rest of my field?',
    'What conditions help this disease spread faster?',
    'How quickly can this move through my rice crop?',
    'Does weather affect how fast this spreads?',
    'Is this likely to spread to my neighbours field?',
    'What is the main way this disease travels between plants?',
    'How far can this disease reach from one infected plant?',
    'Can rain or wind carry this disease to other rows?',
  ],
  management: [
    'What should I do to treat this?',
    'What management steps do you recommend for this?',
    'How do I manage this disease going forward?',
    'What treatment should I apply?',
    'What is the best approach to handle this infection?',
    'What are the recommended control measures?',
    'What is the most effective treatment for this level of infection?',
    'How should I respond to this disease on my crop?',
    'What practical steps can I take right now?',
    'Give me a management plan for this situation.',
    'What action should I take given what you see here?',
    'What does extension recommend for this disease?',
  ],
  prevention: [
    'How can I prevent this from coming back next season?',
    'What preventive steps should I take for the future?',
    'How do I stop this disease from returning?',
    'What can I do before next planting to prevent this?',
    'What long-term strategy prevents this disease?',
    'How do I break the disease cycle for future seasons?',
    'What cultural practices help prevent this?',
    'What should I do differently next season to avoid this?',
    'Is there a way to prevent this disease entirely?',
    'What pre-season preparations reduce this disease risk?',
  ],
  scouting: [
    'How often should I check my field for this?',
    'What is a good scouting routine for this disease?',
    'How do I monitor this going forward?',
    'What should I look for when scouting my field?',
    'How many plants should I check per field visit?',
    'What time of day is best for scouting this disease?',
    'How frequently should I inspect during the critical period?',
    'What scouting pattern do you recommend?',
    'When is the most important time to be checking?',
    'How do I set up a proper field monitoring schedule?',
  ],
  regional: [
    'Is this disease common in Punjab?',
    'Which districts in Pakistan see this the most?',
    'How widespread is this disease in my region?',
    'Is this a major problem in the rice belt of Punjab?',
    'What areas of Pakistan are most affected by this?',
    'Are some districts at higher risk than others?',
    'How prevalent is this across Pakistani rice-growing areas?',
    'Is my area particularly vulnerable to this disease?',
    'Which farming districts should worry most about this?',
  ],
  variety: [
    'Should I change my rice variety because of this?',
    'Which varieties are more resistant to this disease?',
    'Is Super Basmati a good choice given this disease?',
    'What variety would give me better protection?',
    'Does variety choice make a big difference for this disease?',
    'Can I reduce disease risk by switching varieties?',
    'What resistant varieties are available locally?',
    'Which varieties does the Rice Research Institute recommend?',
    'Is there a variety that resists this disease well?',
  ],
  nutrition: [
    'Does fertilizer management affect this disease?',
    'How does nitrogen relate to this problem?',
    'What role does soil nutrition play in this disease?',
    'Should I change my fertilizer approach because of this?',
    'Can better nutrition help my crop resist this?',
    'What nutrient deficiency makes this disease worse?',
    'How should I adjust my fertilizer plan?',
    'Does a soil test matter for managing this?',
    'What is the connection between feeding my crop and this disease?',
  ],
  water: [
    'How does water management affect this disease?',
    'Should I change my irrigation because of this?',
    'Does flooding help or hurt with this infection?',
    'What water practices reduce this disease risk?',
    'Can this disease spread through my irrigation water?',
    'What is the safest water management with this present?',
    'How does my water management relate to this disease?',
    'Should I drain or flood my field with this infection?',
    'Does shared canal water increase the risk?',
  ],
  seed: [
    'Can this disease come from my seed?',
    'Should I worry about my seed source because of this?',
    'Does seed treatment help prevent this disease?',
    'Is it safe to save seed from this crop?',
    'Can this pathogen survive on stored seed?',
    'What seed precautions should I take?',
    'Should I switch to certified seed because of this?',
    'How important is seed quality for this disease?',
  ],
  economics: [
    'How much yield could I lose from this?',
    'What is the economic impact of this disease level?',
    'How does this affect my harvest?',
    'What yield loss should I expect?',
    'Is the treatment cost justified at this severity?',
    'How much does this disease reduce grain quality?',
    'What is the financial risk if I do nothing?',
    'How does this damage translate to actual crop loss?',
  ],
  mistakes: [
    'What mistakes should I avoid with this disease?',
    'What should I definitely NOT do right now?',
    'What common errors make this disease worse?',
    'What practices would be harmful with this infection?',
    'What should I be careful not to do?',
    'What management mistakes do farmers typically make with this?',
    'What actions would accidentally help this disease spread?',
    'Is there anything that seems helpful but actually makes this worse?',
  ],
  confirm_healthy: [
    'Are you sure this is healthy? How can you tell?',
    'What would early disease look like if it was starting?',
    'Could I be missing early symptoms on this leaf?',
    'How confident are you that there is no disease here?',
    'What signs would tell me disease is about to start?',
    'Is there any chance of hidden infection not visible yet?',
    'Double check: is this leaf definitely clean?',
    'What early warning signs should concern me?',
    'How do I know disease is not hiding in the lower canopy?',
  ],
  cure: [
    'What chemical should I spray to cure this disease?',
    'What fungicide or pesticide do you recommend for this?',
    'What is the best cure for this disease?',
    'Tell me the exact spray and dosage I should use.',
    'What product should I buy from the market for this?',
    'What chemical treatment will stop this disease?',
    'Which fungicide works best for this and how much per acre?',
    'What should I spray and how much water should I mix it in?',
    'Give me the name and dose of the chemical to use.',
    'What medicine should I apply on my crop for this?',
    'Is there a specific spray that cures this disease?',
    'What chemical do farmers in Punjab use for this?',
    'What is the recommended spray for this level of infection?',
    'How many sprays will I need and what product should I use?',
    'My crop has this disease. What exact treatment do I apply?',
  ],
  general_care: [
    'What general care does my healthy crop need?',
    'How do I keep my rice field in good shape?',
    'What routine maintenance protects a healthy crop?',
    'What should I focus on to maintain this health?',
    'What are the pillars of good rice crop management?',
    'How do I protect this healthy crop through the season?',
    'What baseline management keeps rice disease-free?',
  ],
}

const CATEGORY_KEYWORDS: Record<SuggestionCategory, string[]> = {
  visual_detail: ['visual', 'look', 'mark', 'lesion', 'pattern', 'symptom', 'spot', 'detail'],
  severity: ['severity', 'severe', 'mild', 'advanced', 'damage', 'urgent'],
  spread: ['spread', 'move', 'travel', 'rain', 'wind', 'infect'],
  management: ['manage', 'management', 'control', 'recommend', 'step', 'action', 'plan'],
  prevention: ['prevent', 'prevention', 'future', 'next season', 'returning'],
  scouting: ['scout', 'monitor', 'check', 'inspect', 'field visit'],
  regional: ['punjab', 'pakistan', 'district', 'region', 'area'],
  variety: ['variety', 'super basmati', 'resistant', 'rice research'],
  nutrition: ['fertilizer', 'nitrogen', 'nutrition', 'soil', 'nutrient'],
  water: ['water', 'irrigation', 'flood', 'drain', 'canal'],
  seed: ['seed', 'certified seed', 'seed treatment', 'stored seed'],
  economics: ['yield', 'loss', 'economic', 'financial', 'harvest', 'cost'],
  mistakes: ['mistake', 'avoid', 'not do', 'harmful', 'careful'],
  confirm_healthy: ['healthy', 'clean', 'early symptom', 'hidden infection', 'warning sign'],
  cure: ['cure', 'spray', 'fungicide', 'pesticide', 'chemical', 'dosage', 'medicine'],
  general_care: ['general care', 'routine', 'maintenance', 'disease-free', 'baseline'],
}

const DEFAULT_CATEGORY_ORDER: SuggestionCategory[] = ['management', 'severity', 'prevention', 'spread']

function hashText(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0
  }
  return hash
}

function sortByRelevance(messages: MessageLike[]): SuggestionCategory[] {
  const recentContext = messages
    .slice(-6)
    .map((m) => m.content.toLowerCase())
    .join(' ')
  const scores = new Map<SuggestionCategory, number>()

  ;(Object.keys(CATEGORY_KEYWORDS) as SuggestionCategory[]).forEach((category) => {
    let score = 0
    for (const keyword of CATEGORY_KEYWORDS[category]) {
      if (recentContext.includes(keyword)) {
        score += 1
      }
    }
    if (score > 0) scores.set(category, score)
  })

  const ranked = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([category]) => category)

  return ranked.length > 0 ? ranked : DEFAULT_CATEGORY_ORDER
}

function pickFromCategory(
  category: SuggestionCategory,
  seed: number,
  used: Set<string>,
  history: Set<string>,
): string | null {
  const pool = SUGGESTION_POOLS[category]
  if (pool.length === 0) return null

  for (let i = 0; i < pool.length; i += 1) {
    const index = (seed + i) % pool.length
    const candidate = pool[index]
    const normalized = candidate.toLowerCase()
    if (used.has(normalized) || history.has(normalized)) continue
    used.add(normalized)
    return candidate
  }
  return null
}

export function getNextPromptSuggestions(messages: MessageLike[], count = 2): string[] {
  const userTurns = messages.filter((m) => m.role === 'user').length
  if (userTurns < 3) return []

  const rankedCategories = sortByRelevance(messages)
  const conversationSeed = hashText(
    messages
      .slice(-6)
      .map((m) => `${m.role}:${m.content}`)
      .join('|'),
  )

  const used = new Set<string>()
  const priorUserMessages = new Set(
    messages.filter((m) => m.role === 'user').map((m) => m.content.trim().toLowerCase()),
  )
  const suggestions: string[] = []

  for (let i = 0; i < rankedCategories.length && suggestions.length < count; i += 1) {
    const picked = pickFromCategory(rankedCategories[i], conversationSeed + i * 7, used, priorUserMessages)
    if (picked) suggestions.push(picked)
  }

  if (suggestions.length < count) {
    for (const category of DEFAULT_CATEGORY_ORDER) {
      if (suggestions.length >= count) break
      const picked = pickFromCategory(
        category,
        conversationSeed + suggestions.length * 11,
        used,
        priorUserMessages,
      )
      if (picked) suggestions.push(picked)
    }
  }

  return suggestions.slice(0, count)
}
