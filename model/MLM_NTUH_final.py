import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np
import torch

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict=None):
        super(BertEmbeddings, self).__init__()

        if feature_dict is None:
            self.feature_dict = {
                'word':True,
                'med':True, # OWN EMBEDDINGS
                'triage':True, # OWN EMBEDDINGS
                'lab': True, # OWN EMBEDDINGS

                'admin': True, # OWN EMBEDDINGS
                'admin_ext': True, # OWN EMBEDDINGS
                'scan1': True, # OWN EMBEDDINGS
                'scan2': True, # OWN EMBEDDINGS
                'scan3': True, # OWN EMBEDDINGS
                'scan4': True, # OWN EMBEDDINGS
                'indicator': True, # OWN EMBEDDINGS
                'gcs': True, # OWN EMBEDDINGS


                'seg':True,
                'age':True,
                'position': True,
            }
        else:
            self.feature_dict = feature_dict

        if feature_dict['word']:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['med']:
            self.med_embeddings = nn.Embedding(config.med_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['triage']:
            self.triage_embeddings = nn.Embedding(config.triage_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['lab']:
            self.lab_embeddings = nn.Embedding(config.lab_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['admin']:
            self.admin_embeddings = nn.Embedding(config.admin_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['admin_ext']:
            self.admin_ext_embeddings = nn.Embedding(config.admin_ext_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['scan1']:
            self.scan1_embeddings = nn.Embedding(config.scan1_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['scan2']:
            self.scan2_embeddings = nn.Embedding(config.scan2_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['scan3']:
            self.scan3_embeddings = nn.Embedding(config.scan3_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['scan4']:
            self.scan4_embeddings = nn.Embedding(config.scan4_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['indicator']:
            self.indicator_embeddings = nn.Embedding(config.indicator_vocab_size, config.hidden_size)

        # OWN EMBEDDING
        if feature_dict['gcs']:
            self.gcs_embeddings = nn.Embedding(config.gcs_vocab_size, config.hidden_size)


        if feature_dict['seg']:
            self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)

        if feature_dict['age']:
            self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)

        if feature_dict['position']:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))


        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, med_input_ids, triage_input_ids, 
                lab_input_ids, 
                admin_input_ids, admin_ext_input_ids,
                scan1_input_ids, scan2_input_ids, scan3_input_ids, scan4_input_ids,
                indicator_input_ids, gcs_input_ids,
                age_ids, seg_ids, posi_ids):
        embeddings = self.word_embeddings(word_ids)
        
        if self.feature_dict['med']:
            med_embed = self.med_embeddings(med_input_ids)
            embeddings = embeddings + med_embed

        if self.feature_dict['triage']:
            triage_embed = self.triage_embeddings(triage_input_ids)
            embeddings = embeddings + triage_embed

        if self.feature_dict['lab']:
            lab_embed = self.lab_embeddings(lab_input_ids)
            embeddings = embeddings + lab_embed

        if self.feature_dict['admin']:
            admin_embed = self.admin_embeddings(admin_input_ids)
            embeddings = embeddings + admin_embed

        if self.feature_dict['admin_ext']:
            admin_ext_embed = self.admin_ext_embeddings(admin_ext_input_ids)
            embeddings = embeddings + admin_ext_embed

        if self.feature_dict['scan1']:
            scan1_embed = self.scan1_embeddings(scan1_input_ids)
            embeddings = embeddings + scan1_embed

        if self.feature_dict['scan2']:
            scan2_embed = self.scan2_embeddings(scan2_input_ids)
            embeddings = embeddings + scan2_embed

        if self.feature_dict['scan3']:
            scan3_embed = self.scan3_embeddings(scan3_input_ids)
            embeddings = embeddings + scan3_embed

        if self.feature_dict['scan4']:
            scan4_embed = self.scan4_embeddings(scan4_input_ids)
            embeddings = embeddings + scan4_embed

        if self.feature_dict['indicator']:
            indicator_embed = self.indicator_embeddings(indicator_input_ids)
            embeddings = embeddings + indicator_embed

        if self.feature_dict['gcs']:
            gcs_embed = self.gcs_embeddings(gcs_input_ids)
            embeddings = embeddings + gcs_embed


        if self.feature_dict['seg']:
            segment_embed = self.segment_embeddings(seg_ids)
            embeddings = embeddings + segment_embed

        if self.feature_dict['age']:
            age_embed = self.age_embeddings(age_ids)
            embeddings = embeddings + age_embed

        if self.feature_dict['position']:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings = embeddings + posi_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, med_input_ids, triage_input_ids, 
                lab_input_ids,
                admin_input_ids, admin_ext_input_ids,
                scan1_input_ids, scan2_input_ids, scan3_input_ids, scan4_input_ids,
                indicator_input_ids, gcs_input_ids,

                age_ids, seg_ids, posi_ids, attention_mask,
                output_all_encoded_layers=True):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        # OWN EMBEDDINGS
        embedding_output = self.embeddings(input_ids, med_input_ids, triage_input_ids, 
                                           lab_input_ids,
                                            admin_input_ids, admin_ext_input_ids,
                                            scan1_input_ids, scan2_input_ids, scan3_input_ids, scan4_input_ids,
                                            indicator_input_ids, gcs_input_ids,

                                           age_ids, seg_ids, posi_ids,)


        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


mlm_weight = 0.999
los_weight = 0.001

class BertForMultiTask(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, los_num_labels, feature_dict, weights=None):
        super(BertForMultiTask, self).__init__(config)
        self.los_num_labels = los_num_labels
        self.bert = BertModel(config, feature_dict)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier2 = nn.Linear(config.hidden_size, los_num_labels)
        self.apply(self.init_bert_weights)

        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float)
        else:
            self.weights = None

    def forward(self, input_ids, med_input_ids, triage_input_ids, 
                lab_input_ids,
                admin_input_ids, admin_ext_input_ids,
                scan1_input_ids, scan2_input_ids, scan3_input_ids, scan4_input_ids,
                indicator_input_ids, gcs_input_ids,
                age_ids=None, 
                seg_ids=None, posi_ids=None, attention_mask=None, 
                masked_lm_labels=None, los_labels=None):
        
        sequence_output, _ = self.bert(input_ids, med_input_ids, triage_input_ids, 
                                                    lab_input_ids,
                                                    admin_input_ids, admin_ext_input_ids,
                                                    scan1_input_ids, scan2_input_ids, scan3_input_ids, scan4_input_ids,
                                                    indicator_input_ids, gcs_input_ids,
                                                   age_ids, seg_ids, posi_ids, attention_mask,
                                                   output_all_encoded_layers=False,)

        # Task 1: masked language model
        dropped_out = self.dropout1(sequence_output)
        prediction_scores = self.classifier1(dropped_out)

        if masked_lm_labels is not None:
            loss_fct1 = nn.CrossEntropyLoss(ignore_index=-1)
            pred1 = prediction_scores.view(-1, self.config.vocab_size)
            label1 = masked_lm_labels.view(-1)
            masked_lm_loss = loss_fct1(pred1, label1)
        else:
            masked_lm_loss = None

        # Task 2: multi-label prediction
        dropped_out_los = self.dropout2(sequence_output)
        logits_los = self.classifier2(dropped_out_los)

        if los_labels is not None:
            if self.weights is not None:
                self.weights = self.weights.to(input_ids.device)
                loss_fct2 = nn.CrossEntropyLoss(weight=self.weights,ignore_index=5)
            else:
                loss_fct2 = nn.CrossEntropyLoss()

            logits_los = logits_los.view(-1, self.los_num_labels)
            loss2 = loss_fct2(logits_los, los_labels.view(-1))
        else:
            loss2 = None


        total_loss = mlm_weight * masked_lm_loss + los_weight * loss2

        return total_loss, prediction_scores, logits_los
