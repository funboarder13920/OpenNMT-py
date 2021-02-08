""" Translation main class """
import os
import torch
from onmt.constants import DefaultTokens
from onmt.inputters.text_dataset import TextMultiField
from onmt.utils.alignment import build_align_pharaoh


def score_exact_match_at_k(pred, tgt, k=5):
    assert len(tgt) >= k, f"tgt too short for exact match at {k}"
    max_k = min([len(pred), len(tgt), k])
    # score = 0
    # for i in range(max_k):
    #     score += pred[i] == tgt[i]
    return 1 if " ".join(pred[:max_k]) == " ".join(tgt[:max_k]) else 0


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(
        self,
        data,
        fields,
        n_best=1,
        replace_unk=False,
        has_tgt=False,
        phrase_table="",
        tokenizer=None,
        exact_match_tokenizer=None,
        stop_at_k=None
    ):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField
        )
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table_dict = {}
        if phrase_table != "" and os.path.exists(phrase_table):
            with open(phrase_table) as phrase_table_fd:
                for line in phrase_table_fd:
                    phrase_src, phrase_trg = line.rstrip("\n").split(
                        DefaultTokens.PHRASE_TABLE_SEPARATOR
                    )
                    self.phrase_table_dict[phrase_src] = phrase_trg
        self.has_tgt = has_tgt
        self.tokenizer = tokenizer
        self.exact_match_tokenizer = exact_match_tokenizer
        self.stop_at_k = stop_at_k

    def maybe_detokenize(self, sequence):
        if self.tokenizer:
            return self.tokenizer._detokenize(sequence).split()
        return sequence

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        vocab = tgt_field.vocab
        tokens = []

        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i][: len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table_dict:
                        src_tok = src_raw[max_index.item()]
                        if src_tok in self.phrase_table_dict:
                            tokens[i] = self.phrase_table_dict[src_tok]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert len(translation_batch["gold_score"]) == len(
            translation_batch["predictions"]
        )
        batch_size = batch.batch_size

        preds, pred_score, attn, align, gold_score, indices = list(
            zip(
                *sorted(
                    zip(
                        translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["alignment"],
                        translation_batch["gold_score"],
                        batch.indices.data,
                    ),
                    key=lambda x: x[-1],
                )
            )
        )

        if not any(align):  # when align is a empty nested list
            align = [None] * batch_size

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = (
            batch.tgt[:, :, 0].index_select(1, perm) if self.has_tgt else None
        )

        translations = []
        for b in range(batch_size):
            if self._has_text_src:
                src_vocab = (
                    self.data.src_vocabs[inds[b]]
                    if self.data.src_vocabs
                    else None
                )
                src_raw = self.data.examples[inds[b]].src[0]
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [
                self.maybe_detokenize(
                    self._build_target_tokens(
                        src[:, b] if src is not None else None,
                        src_vocab,
                        src_raw,
                        preds[b][n],
                        align[b][n] if align[b] is not None else attn[b][n],
                    )
                )
                for n in range(self.n_best)
            ]
            gold_sent = None
            if tgt is not None:
                gold_sent = self.exact_match_tokenizer.detokenize(
                    self.exact_match_tokenizer.tokenize(
                        " ".join(
                            self.maybe_detokenize(
                                self._build_target_tokens(
                                    src[:, b] if src is not None else None,
                                    src_vocab,
                                    src_raw,
                                    tgt[:, b]
                                    if tgt is not None
                                    else None,  # dangerous
                                    None,
                                )
                            )
                        )
                    )[: self.stop_at_k]
                )
            score = 0
            if self.stop_at_k:
                _tgt = batch.tgt[:, :, 0].index_select(1, perm)
                current_target = self.maybe_detokenize(
                    self._build_target_tokens(
                        None,
                        src_vocab,
                        None,
                        _tgt[:, b] if _tgt is not None else None,
                        None,
                    )
                )
                current_target = self.exact_match_tokenizer.tokenize(
                    " ".join(current_target)
                )
                score = 0
                for pred_sent in pred_sents:
                    exact_match_tok_pred_sents = (
                        self.exact_match_tokenizer.tokenize(
                            " ".join(pred_sent)
                        )
                    )
                    score = max(
                        score,
                        score_exact_match_at_k(
                            exact_match_tok_pred_sents,
                            current_target,
                            self.stop_at_k,
                        ),
                    )
                    if score == 1:
                        break
            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw,
                pred_sents,
                attn[b],
                pred_score[b],
                gold_sent,
                score,
                align[b],
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
        word_aligns (List[FloatTensor]): Words Alignment distribution for
            each translation.
    """

    __slots__ = [
        "src",
        "src_raw",
        "pred_sents",
        "attns",
        "pred_scores",
        "gold_sent",
        "gold_score",
        "word_aligns",
    ]

    def __init__(
        self,
        src,
        src_raw,
        pred_sents,
        attn,
        pred_scores,
        tgt_sent,
        gold_score,
        word_aligns,
    ):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        self.word_aligns = word_aligns

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = [" ".join(self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = " ".join(best_pred)
        msg.append("{}".format(pred_sent))
        msg.append("{:.4f}".format(best_score))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh = build_align_pharaoh(pred_align)
            pred_align_sent = " ".join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = " ".join(self.gold_sent)
            msg.append("{}".format(tgt_sent))
            msg.append(("{:.4f}".format(self.gold_score)))
        # if len(self.pred_sents) > 1:
        #     msg.append("\nBEST HYP:\n")
        #     for score, sent in zip(
        #         self.pred_scores, [" ".join(sent) for sent in self.pred_sents]
        #     ):
        #         msg.append("[{:.4f}] {}\n".format(score, sent))

        return "\t".join(msg) + "\n"
