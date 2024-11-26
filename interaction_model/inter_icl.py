from model_train_icl import *
from Param import *
from inter_fix_icl_loss import *



def main():
    print("----------------interaction model--------------------")
    cuda_num = CUDA_NUM
    print("GPU num {}".format(cuda_num))
    #print("ko~ko~da~yo~")

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))
    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill), len(set(train_ill) & set(test_ill) )))


    #(candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))

    #interaction features
    nei_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH, "rb")) #neighbor-view interaction similarity feature
    att_features = pickle.load(open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH,'rb')) #attribute-view interaction similarity feature
    des_features = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb")) #description/name-view interaction similarity feature
    onto_features = pickle.load(open('zh_box2_onto_entity_pair_features.pkl', "rb"))
    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH, "rb"))
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    all_features = [] #[nei-view cat att-view cat des/name-view]
    print(len(entity_pairs))
    print(len(nei_features))
    print(len(att_features))
    print(len(des_features))
    print(len(onto_features))
    # Print the shape of each feature
    print("nei_features shape:", np.array(nei_features).shape)
    print("att_features shape:", np.array(att_features).shape)
    print("des_features shape:", np.array(des_features).shape)
    print("onto_features shape:", np.array(onto_features).shape)
    # print(train_ill)
    # print(len(train_ill))
    for i in range(len(entity_pairs)):
        all_features.append(nei_features[i]+ att_features[i]+ des_features[i]+onto_features[i])# 42 concat 42 concat 1.
    print("All features embedding shape: ", np.array(all_features).shape)



    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx,neg_num=NEG_NUM, batch_size=512)
    # print(Train_gene)
    Model = MlP(42 * 2 + 2,11).cuda(cuda_num)
    Optimizer = optim.Adam(Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)

    # ICL and IAL losses
    icl_criterion = icl_loss(tau=0.05, ab_weight=0.5, n_view=2)
    ial_criterion = ial_loss(tau=0.05, ab_weight=0.5, zoom=0.1, reduction="mean")

    # train
    train(Model, Optimizer, Criterion, icl_criterion, ial_criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)

    #save
    # torch.save(Model, open('box2_interaction_model_onto_ja', "wb"))


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()
