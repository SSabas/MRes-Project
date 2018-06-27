\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.20722 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.956291 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.897962 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.11484 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.29881 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.970426 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.0698 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.07706 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.797575 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.686579 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.25937 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.797056 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.0753 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.23153 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.788915 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.685335 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.819337 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.85131 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.11618 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.37291 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.852312 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.06856 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.3564 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.880246 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.925181 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.905585 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.03062 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.735894 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.710822 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.09952 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.16026 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.14116 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.681692 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.906165 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.37683 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.824353 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.17649 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.944838 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.3552 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.943946 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.834579 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24613 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.38555 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.971057 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.814486 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.34659 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.7992 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.644227 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.62873 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.868815 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.03352 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.982382 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.84634 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.536754 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.894557 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.141 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.42121 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.08533 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.763121 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1324 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.22528 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.95549 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.711571 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.972322 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.0856 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.828984 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.576971 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.21404 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.26472 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.23326 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.599725 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.78498 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.60683 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.78272 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.04091 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.19071 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.47644 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.972545 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.860735 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.22131 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.53741 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.874423 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.75736 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.01283 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.660101 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.62061 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.63538 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.606335 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.917313 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.13751 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.866638 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.439659 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.805581 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.18233 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.69525 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.3042 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.721791 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.10269 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.37779 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.727519 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.767436 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.013 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.933311 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.762743 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.682445 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.29694 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.23037 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.35788 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.541367 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.829431 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.88713 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.772433 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.27461 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.25522 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21689 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.974834 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.91291 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.12693 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.30374 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.906386 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01051 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.952477 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.744756 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.699454 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.46289 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.841184 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.09037 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.07362 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.797151 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.587418 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.886722 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.04095 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.34579 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.2996 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.866438 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.0347 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.28068 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.941781 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.867001 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.05165 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.957898 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.764595 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.700504 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.09658 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.23892 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.19497 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.695775 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.965508 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.63232 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.859563 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.29603 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.10351 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.37222 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.949616 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.786664 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.22015 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.31621 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.967784 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.585529 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.48869 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.235 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.808587 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.12553 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.836393 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.900087 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.14732 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.42077 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.850368 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.0679 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.75232 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.21115 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.657469 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.792732 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.02491 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.902786 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.630781 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.627155 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.693758 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.11931 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.36434 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.89387 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.59117 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.930402 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.39574 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.705972 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.615363 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02532 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.13471 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.667796 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 2.03949 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.24606 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.999339 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.830448 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.96694 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.14026 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.889282 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.940096 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.857718 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.892293 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.840508 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.48392 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.793134 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.11297 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.15493 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.09197 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.838805 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02312 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.25847 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.21606 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.28427 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.978144 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.06875 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.16053 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.829807 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.941351 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.06408 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00649 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.02455 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.851312 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.11572 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13836 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.22323 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.850021 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.91618 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.58647 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.900141 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.23451 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.05028 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.34723 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.954839 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.793897 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.15247 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.32682 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.932209 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.795978 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.16903 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.989424 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.859331 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.28492 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.790018 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.04916 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.23207 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.15113 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.708891 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.863022 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.35514 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.34938 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.00735 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.822858 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.02058 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.12825 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.677198 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.814187 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.880661 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.953407 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.04861 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.86335 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.51945 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.14751 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.3178 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.685385 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.748053 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.35294 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.01247 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.05631 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.40557 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.53808 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.865045 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.743782 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.17086 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.36654 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.10138 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.38873 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.51649 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.0517 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.901324 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.17388 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.816502 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.832204 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.01318 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.34008 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.784032 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.20251 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.73786 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.31549 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.754988 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.848594 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.958323 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.00151 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.801343 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.537905 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.876203 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.19993 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.16514 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.724403 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.87848 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.05485 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.40441 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.602021 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.597028 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.25267 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.003 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.670585 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 2.04201 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19788 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.995696 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.961908 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.05249 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.23454 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.922741 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.877758 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.871788 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.748276 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.796082 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.2661 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.805902 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.984811 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.05625 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.02165 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.696936 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.970469 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.05977 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.49 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.37108 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.90725 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.09756 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.17989 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.837819 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.904441 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.05494 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.954475 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.924814 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.781298 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.16557 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.17961 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.17283 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.779095 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.862371 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.4828 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.873853 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.2293 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.02995 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.16769 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.964989 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.893246 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.15602 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14554 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.982473 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.93885 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11075 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00135 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.83375 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.17454 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.875997 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.991893 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.1346 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.95748 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.775745 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.967179 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.09769 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.23691 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.09842 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.798242 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.02873 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11257 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.955429 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.880739 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.00252 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.980434 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.02531 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.845372 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.22775 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.04866 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.10119 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.780799 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.809209 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.19919 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.940843 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.931257 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.16697 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.183000165432007 z_k1_s0000
                                 - 0.0719999764502963 z_k1_s0001
                                 - 0.0699999883127883 z_k1_s0002
                                 - 0.0669999657565255 z_k1_s0003
                                 - 0.146999894756854 z_k1_s0004
                                 - 0.0430000274064278 z_k1_s0005
                                 - 0.0990003408566645 z_k1_s0006
                                 - 0.126000140363022 z_k1_s0007
                                 - 0.107000060101294 z_k1_s0010
                                 - 0.0930000082300822 z_k1_s0011
                                 - 0.344000190777992 z_k1_s0012
                                 - 0.097000004364708 z_k1_s0013
                                 - 0.388000017458832 z_k1_s0014
                                 - 0.0939999745637277 z_k1_s0015
                                 - 0.0890000120954564 z_k1_s0016
                                 - 0.0960000380310625 z_k1_s0017
                                 - 0.148999603067242 z_k1_s0020
                                 - 0.1350002689131 z_k1_s0021
                                 - 0.0320000150090445 z_k1_s0022
                                 - 0.0410000071365916 z_k1_s0023
                                 - 0.036999976235469 z_k1_s0024
                                 - 0.129000222561416 z_k1_s0025
                                 - 0.134000261187819 z_k1_s0026
                                 - 0.117000129858048 z_k1_s0027
                                 - 0.157000147512279 z_k1_s0030
                                 - 0.0340000138306879 z_k1_s0031
                                 - 0.148999976963912 z_k1_s0032
                                 - 0.0329999925121421 z_k1_s0033
                                 - 0.0839996577582384 z_k1_s0034
                                 - 0.161000232786462 z_k1_s0035
                                 - 0.0429999923976393 z_k1_s0036
                                 - 0.049999999427486 z_k1_s0037
                                 - 0.202999893153902 z_k1_s0040
                                 - 0.136999834847376 z_k1_s0041
                                 - 0.351998645904358 z_k1_s0042
                                 - 0.22200035222168 z_k1_s0043
                                 - 0.188999308420176 z_k1_s0044
                                 - 0.101999777910627 z_k1_s0045
                                 - 0.219999153690224 z_k1_s0046
                                 - 0.135999703880836 z_k1_s0047
                                 - 0.0899999984055396 z_k1_s0050
                                 - 0.0749999210045988 z_k1_s0051
                                 - 0.0319999942553018 z_k1_s0052
                                 - 0.0780000296848081 z_k1_s0053
                                 - 0.0449999992027698 z_k1_s0054
                                 - 0.0589998073769214 z_k1_s0055
                                 - 0.0440000095760439 z_k1_s0056
                                 - 0.0430000199493181 z_k1_s0057
                                 - 0.32899937277425 z_k1_s0060
                                 - 0.108999940385836 z_k1_s0061
                                 - 0.178999934554686 z_k1_s0062
                                 - 0.109999946893955 z_k1_s0063
                                 - 0.144999867078464 z_k1_s0064
                                 - 0.23900032504184 z_k1_s0065
                                 - 0.246999454307801 z_k1_s0066
                                 - 0.179999633463141 z_k1_s0067
                                 - 0.153999924811582 z_k1_s0070
                                 - 0.0539998920145892 z_k1_s0071
                                 - 0.0909998216616164 z_k1_s0072
                                 - 0.168999949371098 z_k1_s0073
                                 - 0.175999633499427 z_k1_s0074
                                 - 0.0759998953019343 z_k1_s0075
                                 - 0.0959998625813994 z_k1_s0076
                                 - 0.165999944459194 z_k1_s0077
                                 - 0.0910000468827069 z_k1_s0100
                                 - 0.108999797793487 z_k1_s0101
                                 - 0.222000154900683 z_k1_s0102
                                 - 0.118000364749104 z_k1_s0103
                                 - 0.111999986778693 z_k1_s0104
                                 - 0.0930003265395612 z_k1_s0105
                                 - 0.130000198689625 z_k1_s0106
                                 - 0.0469999796339109 z_k1_s0107
                                 - 0.0680001690435906 z_k1_s0110
                                 - 0.0939998167661765 z_k1_s0111
                                 - 0.0560000200947321 z_k1_s0112
                                 - 0.0560000200947321 z_k1_s0113
                                 - 0.0789999680800738 z_k1_s0114
                                 - 0.182000234057865 z_k1_s0115
                                 - 0.0510000255327018 z_k1_s0116
                                 - 0.0890000922041226 z_k1_s0117
                                 - 0.208999960162875 z_k1_s0120
                                 - 0.207000672062653 z_k1_s0121
                                 - 0.101000272653617 z_k1_s0122
                                 - 0.101000272653617 z_k1_s0123
                                 - 0.232000034836503 z_k1_s0124
                                 - 0.138000448041769 z_k1_s0125
                                 - 0.0590000976103716 z_k1_s0126
                                 - 0.224000340429131 z_k1_s0127
                                 - 0.0430000306361361 z_k1_s0130
                                 - 0.0620001068986764 z_k1_s0131
                                 - 0.054000081240771 z_k1_s0132
                                 - 0.128000042726125 z_k1_s0133
                                 - 0.114000334975122 z_k1_s0134
                                 - 0.103000039170246 z_k1_s0135
                                 - 0.065000131845406 z_k1_s0136
                                 - 0.0440000185183592 z_k1_s0137
                                 - 0.122999861475989 z_k1_s0140
                                 - 0.160000336894855 z_k1_s0141
                                 - 0.328999694903975 z_k1_s0142
                                 - 0.223999433254284 z_k1_s0143
                                 - 0.0849999274882519 z_k1_s0144
                                 - 0.141999698670044 z_k1_s0145
                                 - 0.073999912439219 z_k1_s0146
                                 - 0.160999795463726 z_k1_s0147
                                 - 0.0780000052648673 z_k1_s0150
                                 - 0.14800060757439 z_k1_s0151
                                 - 0.295000068104308 z_k1_s0152
                                 - 0.128000220685946 z_k1_s0153
                                 - 0.246000247927669 z_k1_s0154
                                 - 0.130000008774779 z_k1_s0155
                                 - 0.124000018008251 z_k1_s0156
                                 - 0.104000007019823 z_k1_s0157
                                 - 0.229999871008713 z_k1_s0160
                                 - 0.716998952173304 z_k1_s0161
                                 - 0.24099997639086 z_k1_s0162
                                 - 0.44799976161871 z_k1_s0163
                                 - 0.163999978815772 z_k1_s0164
                                 - 0.193000054795994 z_k1_s0165
                                 - 0.262000312138583 z_k1_s0166
                                 - 0.211999900410639 z_k1_s0167
                                 - 0.249999028872367 z_k1_s0170
                                 - 0.253999908868086 z_k1_s0171
                                 - 0.119999727338252 z_k1_s0172
                                 - 0.142999328427284 z_k1_s0173
                                 - 0.142999328427284 z_k1_s0174
                                 - 0.265999784862153 z_k1_s0175
                                 - 0.147999391924524 z_k1_s0176
                                 - 0.0579999073700574 z_k1_s0177
                                 - 0.382999777287984 z_k1_s1000
                                 - 0.440999571692577 z_k1_s1001
                                 - 0.753997327259708 z_k1_s1002
                                 - 0.244999030654276 z_k1_s1003
                                 - 0.609996198365026 z_k1_s1004
                                 - 0.589996143157037 z_k1_s1005
                                 - 0.229999172097573 z_k1_s1006
                                 - 0.403998189612774 z_k1_s1007
                                 - 0.0519999142054016 z_k1_s1010
                                 - 0.175999427249486 z_k1_s1011
                                 - 0.0509999419437192 z_k1_s1012
                                 - 0.123999433244207 z_k1_s1013
                                 - 0.0909997900095449 z_k1_s1014
                                 - 0.0719999180381918 z_k1_s1015
                                 - 0.0559998830520088 z_k1_s1016
                                 - 0.175999427249486 z_k1_s1017
                                 - 0.153999985839383 z_k1_s1020
                                 - 0.184000039480805 z_k1_s1021
                                 - 0.160999985195718 z_k1_s1022
                                 - 0.577998875252023 z_k1_s1023
                                 - 0.278999156545614 z_k1_s1024
                                 - 0.27699949512942 z_k1_s1025
                                 - 0.170000040768133 z_k1_s1026
                                 - 0.170999871476231 z_k1_s1027
                                 - 0.200999359228122 z_k1_s1030
                                 - 0.301998215416403 z_k1_s1031
                                 - 0.467997904258731 z_k1_s1032
                                 - 0.14899951182189 z_k1_s1033
                                 - 0.298997858390459 z_k1_s1034
                                 - 0.385998580178081 z_k1_s1035
                                 - 0.197999243001297 z_k1_s1036
                                 - 0.404997630687473 z_k1_s1037
                                 - 0.255999931304966 z_k1_s1040
                                 - 0.282999666821452 z_k1_s1041
                                 - 0.188000109611492 z_k1_s1042
                                 - 0.744000029011353 z_k1_s1043
                                 - 0.227999961684115 z_k1_s1044
                                 - 0.357998932145027 z_k1_s1045
                                 - 0.151999974456076 z_k1_s1046
                                 - 0.229999942092743 z_k1_s1047
                                 - 0.129999728286272 z_k1_s1050
                                 - 0.0849998278256275 z_k1_s1051
                                 - 0.082999867107302 z_k1_s1052
                                 - 0.367999902156625 z_k1_s1053
                                 - 0.246000159343349 z_k1_s1054
                                 - 0.21699965942992 z_k1_s1055
                                 - 0.197999676106591 z_k1_s1056
                                 - 0.0989998380532953 z_k1_s1057
                                 - 0.138999669831704 z_k1_s1060
                                 - 0.335999602176239 z_k1_s1061
                                 - 0.193000087308684 z_k1_s1062
                                 - 0.137999601023042 z_k1_s1063
                                 - 0.164999881461332 z_k1_s1064
                                 - 0.173000145131432 z_k1_s1065
                                 - 0.0829998317353973 z_k1_s1066
                                 - 0.206998756236363 z_k1_s1067
                                 - 0.317999715683528 z_k1_s1070
                                 - 0.220999680485913 z_k1_s1071
                                 - 0.756998413757495 z_k1_s1072
                                 - 0.540999626235012 z_k1_s1073
                                 - 0.187999697998019 z_k1_s1074
                                 - 0.337000947907944 z_k1_s1075
                                 - 0.24599969611003 z_k1_s1076
                                 - 0.569998830792262 z_k1_s1077
                                 - 1.07699836316056 z_k1_s1100
                                 - 0.329000048705663 z_k1_s1101
                                 - 0.838002679495086 z_k1_s1102
                                 - 1.31899888017479 z_k1_s1103
                                 - 1.54300113707955 z_k1_s1104
                                 - 1.37400039494999 z_k1_s1105
                                 - 0.563999534567152 z_k1_s1106
                                 - 0.640999349753692 z_k1_s1107
                                 - 2.18600173760769 z_k1_s1110
                                 - 1.62599847482625 z_k1_s1111
                                 - 1.25400083402726 z_k1_s1112
                                 - 0.680001103030929 z_k1_s1113
                                 - 0.735000808021171 z_k1_s1114
                                 - 1.76600762903234 z_k1_s1115
                                 - 1.65000226772158 z_k1_s1116
                                 - 1.22100256755511 z_k1_s1117
                                 - 0.719000706777553 z_k1_s1120
                                 - 1.03300018157815 z_k1_s1121
                                 - 1.16000495922707 z_k1_s1122
                                 - 1.0579980806235 z_k1_s1123
                                 - 1.57300350361611 z_k1_s1124
                                 - 1.81400609028067 z_k1_s1125
                                 - 0.425000390914238 z_k1_s1126
                                 - 0.615000377978024 z_k1_s1127
                                 - 0.827999199711338 z_k1_s1130
                                 - 2.2259998175924 z_k1_s1131
                                 - 1.27099637650592 z_k1_s1132
                                 - 0.821998870345751 z_k1_s1133
                                 - 0.986002412612247 z_k1_s1134
                                 - 1.69200230281368 z_k1_s1135
                                 - 1.10799401333222 z_k1_s1136
                                 - 0.702999243719268 z_k1_s1137
                                 - 0.494998990297606 z_k1_s1140
                                 - 0.325001396973955 z_k1_s1141
                                 - 0.211999950116493 z_k1_s1142
                                 - 0.495999223676074 z_k1_s1143
                                 - 0.467999163081417 z_k1_s1144
                                 - 0.279000048867975 z_k1_s1145
                                 - 0.368998717621651 z_k1_s1146
                                 - 0.592999202378904 z_k1_s1147
                                 - 1.10800281432573 z_k1_s1150
                                 - 2.32799739584843 z_k1_s1151
                                 - 3.4850047686643 z_k1_s1152
                                 - 0.827001645671448 z_k1_s1153
                                 - 1.81400974863247 z_k1_s1154
                                 - 2.74201137131969 z_k1_s1155
                                 - 0.881001092241164 z_k1_s1156
                                 - 1.99699876390035 z_k1_s1157
                                 - 0.894999347481372 z_k1_s1160
                                 - 0.98100259222355 z_k1_s1161
                                 - 0.565998612831515 z_k1_s1162
                                 - 0.340999958101901 z_k1_s1163
                                 - 0.605997786338833 z_k1_s1164
                                 - 0.202000319313248 z_k1_s1165
                                 - 0.900998959207394 z_k1_s1166
                                 - 0.7939997164249 z_k1_s1167
                                 - 0.254000183238058 z_k1_s1170
                                 - 0.304000368428356 z_k1_s1171
                                 - 0.48299862707953 z_k1_s1172
                                 - 0.382000295028907 z_k1_s1173
                                 - 0.564000672697336 z_k1_s1174
                                 - 0.455999317541458 z_k1_s1175
                                 - 0.853001973649464 z_k1_s1176
                                 - 0.821000504750496 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.801273 w_k1_ADM_s000
                                 + 0.680596 w_k1_AXP_s000
                                 + 0.775725 w_k1_DIS_s000
                                 + 0.803944 w_k1_FDX_s000
                                 + 0.88567 w_k1_IBM_s000
                                 + 0.95306 w_k1_INTC_s000
                                 + 0.833938 w_k1_KO_s000
                                 + 1.23483 w_k1_MAT_s000
                                 + 0.935457 w_k1_MSFT_s000
                                 + 0.832771 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.942705 w_k1_ADM_s000
                                 + 1.39982 w_k1_AXP_s000 + 1.2598 w_k1_DIS_s000
                                 + 1.10509 w_k1_FDX_s000
                                 + 1.37078 w_k1_IBM_s000
                                 + 1.23351 w_k1_INTC_s000
                                 + 1.20311 w_k1_KO_s000 + 2.3857 w_k1_MAT_s000
                                 + 1.14002 w_k1_MSFT_s000
                                 + 1.06188 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.934852 w_k1_ADM_s000
                                 + 0.945214 w_k1_AXP_s000
                                 + 0.877106 w_k1_DIS_s000
                                 + 0.985633 w_k1_FDX_s000
                                 + 0.724531 w_k1_IBM_s000
                                 + 1.26958 w_k1_INTC_s000
                                 + 0.796755 w_k1_KO_s000
                                 + 1.02387 w_k1_MAT_s000
                                 + 1.11744 w_k1_MSFT_s000
                                 + 0.898842 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.976184 w_k1_ADM_s000
                                 + 1.17061 w_k1_AXP_s000
                                 + 1.24681 w_k1_DIS_s000
                                 + 1.17913 w_k1_FDX_s000
                                 + 0.844327 w_k1_IBM_s000
                                 + 1.14663 w_k1_INTC_s000
                                 + 0.966933 w_k1_KO_s000
                                 + 1.65323 w_k1_MAT_s000
                                 + 1.26267 w_k1_MSFT_s000
                                 + 0.970381 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.942367 w_k1_ADM_s000
                                 + 1.00972 w_k1_AXP_s000
                                 + 1.15147 w_k1_DIS_s000
                                 + 0.875759 w_k1_FDX_s000
                                 + 1.18403 w_k1_IBM_s000
                                 + 0.897017 w_k1_INTC_s000
                                 + 0.965223 w_k1_KO_s000
                                 + 1.24592 w_k1_MAT_s000
                                 + 0.968594 w_k1_MSFT_s000
                                 + 0.988346 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.915853 w_k1_ADM_s000
                                 + 1.14913 w_k1_AXP_s000
                                 + 1.27806 w_k1_DIS_s000
                                 + 0.986749 w_k1_FDX_s000
                                 + 1.22245 w_k1_IBM_s000
                                 + 1.07027 w_k1_INTC_s000
                                 + 1.17348 w_k1_KO_s000 + 2.06213 w_k1_MAT_s000
                                 + 1.11335 w_k1_MSFT_s000
                                 + 1.05963 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.45576 w_k1_ADM_s000
                                 + 1.47856 w_k1_AXP_s000
                                 + 1.75372 w_k1_DIS_s000
                                 + 1.42647 w_k1_FDX_s000
                                 + 1.16363 w_k1_IBM_s000
                                 + 1.4516 w_k1_INTC_s000 + 1.21002 w_k1_KO_s000
                                 + 1.41439 w_k1_MAT_s000
                                 + 1.28015 w_k1_MSFT_s000
                                 + 1.23514 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.660797 w_k1_ADM_s000
                                 + 0.707997 w_k1_AXP_s000
                                 + 0.590385 w_k1_DIS_s000
                                 + 0.65873 w_k1_FDX_s000
                                 + 0.854656 w_k1_IBM_s000
                                 + 0.718781 w_k1_INTC_s000
                                 + 0.59763 w_k1_KO_s000
                                 + 0.707586 w_k1_MAT_s000
                                 + 0.658358 w_k1_MSFT_s000
                                 + 0.791927 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.799296 w_k1_ADM_s001
                                 + 0.650654 w_k1_AXP_s001
                                 + 0.654542 w_k1_DIS_s001
                                 + 0.846102 w_k1_FDX_s001
                                 + 0.572268 w_k1_IBM_s001
                                 + 1.04992 w_k1_INTC_s001
                                 + 0.600668 w_k1_KO_s001
                                 + 0.54058 w_k1_MAT_s001
                                 + 0.799446 w_k1_MSFT_s001
                                 + 0.674794 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.794299 w_k1_ADM_s001
                                 + 0.607741 w_k1_AXP_s001
                                 + 0.569693 w_k1_DIS_s001
                                 + 0.704486 w_k1_FDX_s001
                                 + 0.970744 w_k1_IBM_s001
                                 + 0.984378 w_k1_INTC_s001
                                 + 0.843169 w_k1_KO_s001
                                 + 0.974679 w_k1_MAT_s001
                                 + 0.819383 w_k1_MSFT_s001
                                 + 0.828186 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.886407 w_k1_ADM_s001
                                 + 0.994242 w_k1_AXP_s001
                                 + 1.1735 w_k1_DIS_s001 + 1.2274 w_k1_FDX_s001
                                 + 1.00517 w_k1_IBM_s001
                                 + 1.32458 w_k1_INTC_s001
                                 + 1.09848 w_k1_KO_s001 + 1.03521 w_k1_MAT_s001
                                 + 1.0789 w_k1_MSFT_s001 + 1.06287 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.783819 w_k1_ADM_s001
                                 + 0.951821 w_k1_AXP_s001
                                 + 0.71005 w_k1_DIS_s001
                                 + 0.880679 w_k1_FDX_s001
                                 + 0.798196 w_k1_IBM_s001
                                 + 1.02071 w_k1_INTC_s001
                                 + 0.771141 w_k1_KO_s001
                                 + 1.03869 w_k1_MAT_s001
                                 + 0.832086 w_k1_MSFT_s001
                                 + 0.823897 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.02077 w_k1_ADM_s001
                                 + 0.867585 w_k1_AXP_s001
                                 + 1.09299 w_k1_DIS_s001
                                 + 0.984148 w_k1_FDX_s001
                                 + 1.39894 w_k1_IBM_s001
                                 + 1.19535 w_k1_INTC_s001
                                 + 1.46713 w_k1_KO_s001 + 1.58593 w_k1_MAT_s001
                                 + 1.16198 w_k1_MSFT_s001
                                 + 1.08695 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.08849 w_k1_ADM_s001
                                 + 0.89573 w_k1_AXP_s001
                                 + 0.940973 w_k1_DIS_s001
                                 + 1.18061 w_k1_FDX_s001
                                 + 0.640516 w_k1_IBM_s001
                                 + 1.22853 w_k1_INTC_s001
                                 + 0.849489 w_k1_KO_s001
                                 + 0.668136 w_k1_MAT_s001
                                 + 1.0308 w_k1_MSFT_s001
                                 + 0.902883 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.771853 w_k1_ADM_s001
                                 + 0.639117 w_k1_AXP_s001
                                 + 0.802143 w_k1_DIS_s001
                                 + 0.932523 w_k1_FDX_s001
                                 + 1.08817 w_k1_IBM_s001
                                 + 1.04796 w_k1_INTC_s001 + 1.0157 w_k1_KO_s001
                                 + 0.877284 w_k1_MAT_s001
                                 + 1.00042 w_k1_MSFT_s001 + 0.8629 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.864545 w_k1_ADM_s001
                                 + 0.782691 w_k1_AXP_s001
                                 + 0.733613 w_k1_DIS_s001
                                 + 0.895752 w_k1_FDX_s001
                                 + 0.771762 w_k1_IBM_s001
                                 + 0.946758 w_k1_INTC_s001
                                 + 0.950746 w_k1_KO_s001
                                 + 0.863624 w_k1_MAT_s001
                                 + 0.877231 w_k1_MSFT_s001
                                 + 0.994784 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07976 w_k1_ADM_s002
                                 + 1.24042 w_k1_AXP_s002
                                 + 1.59559 w_k1_DIS_s002
                                 + 1.29272 w_k1_FDX_s002
                                 + 1.06368 w_k1_IBM_s002
                                 + 1.3125 w_k1_INTC_s002 + 1.17109 w_k1_KO_s002
                                 + 1.02847 w_k1_MAT_s002
                                 + 1.33354 w_k1_MSFT_s002
                                 + 0.979759 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.923016 w_k1_ADM_s002
                                 + 0.690842 w_k1_AXP_s002
                                 + 0.667519 w_k1_DIS_s002
                                 + 0.718484 w_k1_FDX_s002
                                 + 0.764925 w_k1_IBM_s002
                                 + 0.651381 w_k1_INTC_s002
                                 + 0.702891 w_k1_KO_s002
                                 + 0.763301 w_k1_MAT_s002
                                 + 0.707051 w_k1_MSFT_s002
                                 + 0.708678 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.968153 w_k1_ADM_s002
                                 + 1.00248 w_k1_AXP_s002
                                 + 1.01964 w_k1_DIS_s002
                                 + 1.12626 w_k1_FDX_s002
                                 + 0.890563 w_k1_IBM_s002
                                 + 1.35491 w_k1_INTC_s002
                                 + 0.97103 w_k1_KO_s002
                                 + 0.935921 w_k1_MAT_s002
                                 + 1.10237 w_k1_MSFT_s002
                                 + 0.894571 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.21545 w_k1_ADM_s002
                                 + 1.23158 w_k1_AXP_s002
                                 + 1.09878 w_k1_DIS_s002
                                 + 1.20848 w_k1_FDX_s002
                                 + 0.807713 w_k1_IBM_s002
                                 + 1.19807 w_k1_INTC_s002
                                 + 0.848391 w_k1_KO_s002
                                 + 0.819073 w_k1_MAT_s002
                                 + 1.06742 w_k1_MSFT_s002
                                 + 0.835366 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.33825 w_k1_ADM_s002
                                 + 1.05138 w_k1_AXP_s002
                                 + 1.16034 w_k1_DIS_s002
                                 + 1.30833 w_k1_FDX_s002
                                 + 0.706979 w_k1_IBM_s002
                                 + 1.16115 w_k1_INTC_s002 + 0.9067 w_k1_KO_s002
                                 + 0.681044 w_k1_MAT_s002
                                 + 1.2216 w_k1_MSFT_s002 + 1.01349 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.942755 w_k1_ADM_s002
                                 + 0.912207 w_k1_AXP_s002
                                 + 0.974136 w_k1_DIS_s002
                                 + 0.952268 w_k1_FDX_s002
                                 + 0.989575 w_k1_IBM_s002
                                 + 1.18066 w_k1_INTC_s002
                                 + 0.992479 w_k1_KO_s002
                                 + 1.20982 w_k1_MAT_s002
                                 + 1.0378 w_k1_MSFT_s002
                                 + 0.834943 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.833184 w_k1_ADM_s002
                                 + 1.04328 w_k1_AXP_s002
                                 + 0.959778 w_k1_DIS_s002
                                 + 0.889962 w_k1_FDX_s002
                                 + 1.40424 w_k1_IBM_s002
                                 + 0.870427 w_k1_INTC_s002
                                 + 1.0433 w_k1_KO_s002 + 1.32928 w_k1_MAT_s002
                                 + 0.894622 w_k1_MSFT_s002
                                 + 0.929493 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.83506 w_k1_ADM_s002
                                 + 0.966566 w_k1_AXP_s002
                                 + 0.833738 w_k1_DIS_s002
                                 + 0.998446 w_k1_FDX_s002
                                 + 0.734862 w_k1_IBM_s002
                                 + 1.07429 w_k1_INTC_s002
                                 + 0.853893 w_k1_KO_s002
                                 + 0.777884 w_k1_MAT_s002
                                 + 1.06038 w_k1_MSFT_s002
                                 + 0.824214 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.30538 w_k1_ADM_s003
                                 + 1.15925 w_k1_AXP_s003
                                 + 1.13865 w_k1_DIS_s003
                                 + 1.04276 w_k1_FDX_s003
                                 + 1.33525 w_k1_IBM_s003
                                 + 0.928098 w_k1_INTC_s003
                                 + 1.10151 w_k1_KO_s003 + 1.07414 w_k1_MAT_s003
                                 + 1.00299 w_k1_MSFT_s003 + 1.2305 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.7216 w_k1_ADM_s003
                                 + 0.667713 w_k1_AXP_s003
                                 + 0.456074 w_k1_DIS_s003
                                 + 0.596125 w_k1_FDX_s003
                                 + 0.906933 w_k1_IBM_s003
                                 + 0.686185 w_k1_INTC_s003
                                 + 0.635372 w_k1_KO_s003
                                 + 0.644558 w_k1_MAT_s003
                                 + 0.607895 w_k1_MSFT_s003
                                 + 0.766865 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.07697 w_k1_ADM_s003
                                 + 0.909184 w_k1_AXP_s003
                                 + 0.913785 w_k1_DIS_s003
                                 + 0.903319 w_k1_FDX_s003
                                 + 0.97931 w_k1_IBM_s003
                                 + 0.806759 w_k1_INTC_s003
                                 + 0.813947 w_k1_KO_s003
                                 + 0.859466 w_k1_MAT_s003
                                 + 0.781005 w_k1_MSFT_s003
                                 + 0.897426 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.33138 w_k1_ADM_s003
                                 + 1.32756 w_k1_AXP_s003
                                 + 1.02554 w_k1_DIS_s003
                                 + 0.971224 w_k1_FDX_s003
                                 + 1.89247 w_k1_IBM_s003
                                 + 0.969216 w_k1_INTC_s003
                                 + 1.18948 w_k1_KO_s003 + 1.48158 w_k1_MAT_s003
                                 + 0.884468 w_k1_MSFT_s003
                                 + 0.952473 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.68434 w_k1_ADM_s003
                                 + 1.15864 w_k1_AXP_s003
                                 + 1.71608 w_k1_DIS_s003
                                 + 1.35224 w_k1_FDX_s003 + 1.0428 w_k1_IBM_s003
                                 + 1.10394 w_k1_INTC_s003
                                 + 1.23334 w_k1_KO_s003 + 0.85879 w_k1_MAT_s003
                                 + 1.42398 w_k1_MSFT_s003
                                 + 1.23087 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.968317 w_k1_ADM_s003
                                 + 0.95259 w_k1_AXP_s003 + 1.0134 w_k1_DIS_s003
                                 + 1.05968 w_k1_FDX_s003
                                 + 0.933333 w_k1_IBM_s003
                                 + 1.20599 w_k1_INTC_s003
                                 + 1.01654 w_k1_KO_s003
                                 + 0.798573 w_k1_MAT_s003
                                 + 1.01891 w_k1_MSFT_s003
                                 + 1.07567 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.00521 w_k1_ADM_s003
                                 + 0.874522 w_k1_AXP_s003
                                 + 0.758658 w_k1_DIS_s003
                                 + 0.932467 w_k1_FDX_s003
                                 + 0.659903 w_k1_IBM_s003
                                 + 0.88323 w_k1_INTC_s003
                                 + 0.771742 w_k1_KO_s003
                                 + 0.488924 w_k1_MAT_s003
                                 + 0.755111 w_k1_MSFT_s003
                                 + 0.96766 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.75859 w_k1_ADM_s003
                                 + 0.817695 w_k1_AXP_s003
                                 + 0.606615 w_k1_DIS_s003
                                 + 0.706294 w_k1_FDX_s003
                                 + 0.982136 w_k1_IBM_s003
                                 + 0.96431 w_k1_INTC_s003
                                 + 0.820259 w_k1_KO_s003
                                 + 0.852367 w_k1_MAT_s003
                                 + 0.728963 w_k1_MSFT_s003
                                 + 0.879206 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.03995 w_k1_ADM_s004
                                 + 0.767656 w_k1_AXP_s004
                                 + 0.914048 w_k1_DIS_s004
                                 + 0.993203 w_k1_FDX_s004
                                 + 0.714402 w_k1_IBM_s004
                                 + 0.817707 w_k1_INTC_s004
                                 + 0.904158 w_k1_KO_s004
                                 + 0.968694 w_k1_MAT_s004
                                 + 0.979416 w_k1_MSFT_s004
                                 + 1.00171 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.943463 w_k1_ADM_s004
                                 + 0.599115 w_k1_AXP_s004
                                 + 0.616908 w_k1_DIS_s004
                                 + 0.777526 w_k1_FDX_s004
                                 + 0.569209 w_k1_IBM_s004
                                 + 0.745532 w_k1_INTC_s004
                                 + 0.709807 w_k1_KO_s004
                                 + 0.596166 w_k1_MAT_s004
                                 + 0.85946 w_k1_MSFT_s004
                                 + 0.879846 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.857495 w_k1_ADM_s004
                                 + 1.08954 w_k1_AXP_s004
                                 + 1.12477 w_k1_DIS_s004
                                 + 1.04931 w_k1_FDX_s004
                                 + 1.50377 w_k1_IBM_s004
                                 + 1.28244 w_k1_INTC_s004
                                 + 1.35215 w_k1_KO_s004 + 1.70083 w_k1_MAT_s004
                                 + 1.0452 w_k1_MSFT_s004 + 1.24123 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.10187 w_k1_ADM_s004
                                 + 1.11477 w_k1_AXP_s004
                                 + 0.729132 w_k1_DIS_s004
                                 + 1.05068 w_k1_FDX_s004
                                 + 0.825478 w_k1_IBM_s004
                                 + 0.919876 w_k1_INTC_s004
                                 + 0.927905 w_k1_KO_s004
                                 + 1.13463 w_k1_MAT_s004
                                 + 1.03532 w_k1_MSFT_s004
                                 + 1.06858 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.27209 w_k1_ADM_s004
                                 + 0.949176 w_k1_AXP_s004
                                 + 1.16406 w_k1_DIS_s004
                                 + 1.22999 w_k1_FDX_s004
                                 + 0.842305 w_k1_IBM_s004
                                 + 0.9511 w_k1_INTC_s004 + 1.01637 w_k1_KO_s004
                                 + 1.1305 w_k1_MAT_s004 + 1.1859 w_k1_MSFT_s004
                                 + 1.08165 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.32866 w_k1_ADM_s004
                                 + 1.29959 w_k1_AXP_s004 + 1.7464 w_k1_DIS_s004
                                 + 1.31934 w_k1_FDX_s004
                                 + 1.19282 w_k1_IBM_s004
                                 + 1.1365 w_k1_INTC_s004 + 1.46528 w_k1_KO_s004
                                 + 1.15287 w_k1_MAT_s004
                                 + 1.31414 w_k1_MSFT_s004
                                 + 1.29314 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.877627 w_k1_ADM_s004
                                 + 0.954628 w_k1_AXP_s004
                                 + 0.72845 w_k1_DIS_s004
                                 + 0.796278 w_k1_FDX_s004
                                 + 1.1499 w_k1_IBM_s004
                                 + 0.770124 w_k1_INTC_s004
                                 + 1.02576 w_k1_KO_s004 + 1.36338 w_k1_MAT_s004
                                 + 0.845098 w_k1_MSFT_s004
                                 + 1.03216 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.654592 w_k1_ADM_s004
                                 + 0.644157 w_k1_AXP_s004
                                 + 0.565194 w_k1_DIS_s004
                                 + 0.751042 w_k1_FDX_s004
                                 + 0.913215 w_k1_IBM_s004
                                 + 0.756335 w_k1_INTC_s004
                                 + 0.683091 w_k1_KO_s004
                                 + 0.716572 w_k1_MAT_s004
                                 + 0.741339 w_k1_MSFT_s004
                                 + 0.839617 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.902278 w_k1_ADM_s005
                                 + 1.39651 w_k1_AXP_s005 + 1.2457 w_k1_DIS_s005
                                 + 1.14255 w_k1_FDX_s005
                                 + 1.16276 w_k1_IBM_s005
                                 + 1.25538 w_k1_INTC_s005 + 1.111 w_k1_KO_s005
                                 + 1.58776 w_k1_MAT_s005
                                 + 1.06254 w_k1_MSFT_s005
                                 + 1.08192 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.04429 w_k1_ADM_s005
                                 + 0.994354 w_k1_AXP_s005
                                 + 1.01037 w_k1_DIS_s005
                                 + 1.10499 w_k1_FDX_s005
                                 + 0.699265 w_k1_IBM_s005
                                 + 1.19494 w_k1_INTC_s005
                                 + 0.919473 w_k1_KO_s005
                                 + 0.86393 w_k1_MAT_s005
                                 + 1.08995 w_k1_MSFT_s005
                                 + 0.959015 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.32917 w_k1_ADM_s005
                                 + 1.14975 w_k1_AXP_s005
                                 + 0.872582 w_k1_DIS_s005
                                 + 0.956076 w_k1_FDX_s005
                                 + 0.857967 w_k1_IBM_s005
                                 + 0.88319 w_k1_INTC_s005
                                 + 0.95588 w_k1_KO_s005 + 1.21774 w_k1_MAT_s005
                                 + 0.768326 w_k1_MSFT_s005
                                 + 0.937049 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.799696 w_k1_ADM_s005
                                 + 0.852041 w_k1_AXP_s005
                                 + 0.621459 w_k1_DIS_s005
                                 + 0.753639 w_k1_FDX_s005
                                 + 0.838985 w_k1_IBM_s005
                                 + 1.12847 w_k1_INTC_s005
                                 + 0.735242 w_k1_KO_s005
                                 + 0.648723 w_k1_MAT_s005
                                 + 0.812527 w_k1_MSFT_s005
                                 + 0.8572 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.881708 w_k1_ADM_s005
                                 + 0.943592 w_k1_AXP_s005
                                 + 0.96437 w_k1_DIS_s005
                                 + 0.915543 w_k1_FDX_s005
                                 + 1.21742 w_k1_IBM_s005
                                 + 1.31925 w_k1_INTC_s005
                                 + 1.17447 w_k1_KO_s005
                                 + 0.936912 w_k1_MAT_s005
                                 + 1.05123 w_k1_MSFT_s005
                                 + 1.00723 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.990215 w_k1_ADM_s005
                                 + 0.933289 w_k1_AXP_s005
                                 + 0.895385 w_k1_DIS_s005
                                 + 0.885107 w_k1_FDX_s005
                                 + 1.1841 w_k1_IBM_s005 + 1.0178 w_k1_INTC_s005
                                 + 0.948202 w_k1_KO_s005
                                 + 0.970006 w_k1_MAT_s005
                                 + 0.766347 w_k1_MSFT_s005
                                 + 0.962755 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.2145 w_k1_ADM_s005
                                 + 1.11028 w_k1_AXP_s005
                                 + 1.23869 w_k1_DIS_s005
                                 + 1.09649 w_k1_FDX_s005
                                 + 1.03613 w_k1_IBM_s005
                                 + 1.21875 w_k1_INTC_s005
                                 + 1.07862 w_k1_KO_s005
                                 + 0.965669 w_k1_MAT_s005
                                 + 1.04064 w_k1_MSFT_s005
                                 + 0.919839 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.26238 w_k1_ADM_s005
                                 + 1.57209 w_k1_AXP_s005
                                 + 1.78053 w_k1_DIS_s005
                                 + 1.43263 w_k1_FDX_s005
                                 + 1.26217 w_k1_IBM_s005
                                 + 1.38102 w_k1_INTC_s005
                                 + 1.36987 w_k1_KO_s005 + 1.57482 w_k1_MAT_s005
                                 + 1.20578 w_k1_MSFT_s005
                                 + 1.15092 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.944106 w_k1_ADM_s006
                                 + 1.15364 w_k1_AXP_s006
                                 + 1.39494 w_k1_DIS_s006
                                 + 1.12025 w_k1_FDX_s006 + 1.1449 w_k1_IBM_s006
                                 + 1.12855 w_k1_INTC_s006
                                 + 1.09551 w_k1_KO_s006 + 1.09782 w_k1_MAT_s006
                                 + 1.10203 w_k1_MSFT_s006
                                 + 1.10956 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.81887 w_k1_ADM_s006
                                 + 0.746183 w_k1_AXP_s006
                                 + 1.02167 w_k1_DIS_s006
                                 + 0.909384 w_k1_FDX_s006
                                 + 0.754332 w_k1_IBM_s006
                                 + 0.928111 w_k1_INTC_s006
                                 + 0.833842 w_k1_KO_s006
                                 + 1.07112 w_k1_MAT_s006
                                 + 0.989791 w_k1_MSFT_s006
                                 + 1.0056 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.14323 w_k1_ADM_s006
                                 + 0.965968 w_k1_AXP_s006
                                 + 0.935906 w_k1_DIS_s006
                                 + 1.11838 w_k1_FDX_s006
                                 + 0.78334 w_k1_IBM_s006
                                 + 1.03947 w_k1_INTC_s006
                                 + 0.816439 w_k1_KO_s006
                                 + 0.842822 w_k1_MAT_s006
                                 + 0.956916 w_k1_MSFT_s006
                                 + 1.03423 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04203 w_k1_ADM_s006
                                 + 1.10373 w_k1_AXP_s006
                                 + 1.14539 w_k1_DIS_s006
                                 + 0.968647 w_k1_FDX_s006
                                 + 0.832931 w_k1_IBM_s006
                                 + 0.878888 w_k1_INTC_s006
                                 + 0.97614 w_k1_KO_s006 + 1.48755 w_k1_MAT_s006
                                 + 0.982746 w_k1_MSFT_s006
                                 + 0.9888 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02128 w_k1_ADM_s006
                                 + 1.19565 w_k1_AXP_s006
                                 + 0.823387 w_k1_DIS_s006
                                 + 0.86115 w_k1_FDX_s006
                                 + 1.31052 w_k1_IBM_s006
                                 + 0.860939 w_k1_INTC_s006
                                 + 0.889613 w_k1_KO_s006
                                 + 1.77328 w_k1_MAT_s006
                                 + 0.856642 w_k1_MSFT_s006
                                 + 1.10324 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.948233 w_k1_ADM_s006
                                 + 0.869551 w_k1_AXP_s006
                                 + 1.00739 w_k1_DIS_s006
                                 + 0.950931 w_k1_FDX_s006
                                 + 1.1157 w_k1_IBM_s006
                                 + 0.903481 w_k1_INTC_s006
                                 + 1.12074 w_k1_KO_s006
                                 + 0.940336 w_k1_MAT_s006
                                 + 1.01722 w_k1_MSFT_s006
                                 + 1.00348 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.804527 w_k1_ADM_s006
                                 + 0.87266 w_k1_AXP_s006
                                 + 0.687051 w_k1_DIS_s006
                                 + 0.72048 w_k1_FDX_s006
                                 + 0.764999 w_k1_IBM_s006
                                 + 0.678641 w_k1_INTC_s006
                                 + 0.718713 w_k1_KO_s006
                                 + 0.870458 w_k1_MAT_s006
                                 + 0.70533 w_k1_MSFT_s006
                                 + 0.848387 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.66955 w_k1_ADM_s006
                                 + 1.53189 w_k1_AXP_s006
                                 + 1.76223 w_k1_DIS_s006
                                 + 1.43317 w_k1_FDX_s006
                                 + 0.976992 w_k1_IBM_s006
                                 + 1.17122 w_k1_INTC_s006
                                 + 1.15357 w_k1_KO_s006 + 1.1848 w_k1_MAT_s006
                                 + 1.29816 w_k1_MSFT_s006
                                 + 1.32683 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.974538 w_k1_ADM_s007
                                 + 0.897207 w_k1_AXP_s007
                                 + 1.02142 w_k1_DIS_s007
                                 + 0.956425 w_k1_FDX_s007
                                 + 0.934784 w_k1_IBM_s007
                                 + 0.894199 w_k1_INTC_s007
                                 + 1.10913 w_k1_KO_s007
                                 + 0.996662 w_k1_MAT_s007
                                 + 1.06064 w_k1_MSFT_s007
                                 + 1.03055 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.961093 w_k1_ADM_s007
                                 + 1.35489 w_k1_AXP_s007
                                 + 0.928856 w_k1_DIS_s007
                                 + 1.06831 w_k1_FDX_s007
                                 + 1.35091 w_k1_IBM_s007
                                 + 1.19124 w_k1_INTC_s007
                                 + 1.17122 w_k1_KO_s007 + 1.77831 w_k1_MAT_s007
                                 + 0.979955 w_k1_MSFT_s007
                                 + 1.02278 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.605902 w_k1_ADM_s007
                                 + 0.81972 w_k1_AXP_s007
                                 + 0.895303 w_k1_DIS_s007
                                 + 0.840558 w_k1_FDX_s007
                                 + 1.38032 w_k1_IBM_s007
                                 + 1.07983 w_k1_INTC_s007 + 1.0183 w_k1_KO_s007
                                 + 1.31569 w_k1_MAT_s007
                                 + 0.987532 w_k1_MSFT_s007
                                 + 0.945634 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.68994 w_k1_ADM_s007
                                 + 0.798051 w_k1_AXP_s007
                                 + 0.640602 w_k1_DIS_s007
                                 + 0.748777 w_k1_FDX_s007
                                 + 0.94757 w_k1_IBM_s007
                                 + 0.892022 w_k1_INTC_s007
                                 + 0.802619 w_k1_KO_s007
                                 + 1.02272 w_k1_MAT_s007
                                 + 0.900129 w_k1_MSFT_s007
                                 + 0.772893 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.67418 w_k1_ADM_s007
                                 + 0.84142 w_k1_AXP_s007
                                 + 0.872362 w_k1_DIS_s007
                                 + 0.917772 w_k1_FDX_s007
                                 + 0.752384 w_k1_IBM_s007
                                 + 1.02828 w_k1_INTC_s007
                                 + 0.804725 w_k1_KO_s007
                                 + 0.643253 w_k1_MAT_s007
                                 + 0.930939 w_k1_MSFT_s007
                                 + 0.910672 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.90752 w_k1_ADM_s007
                                 + 1.03286 w_k1_AXP_s007
                                 + 1.12096 w_k1_DIS_s007
                                 + 1.07691 w_k1_FDX_s007
                                 + 1.12722 w_k1_IBM_s007
                                 + 0.864308 w_k1_INTC_s007
                                 + 0.991761 w_k1_KO_s007
                                 + 1.20304 w_k1_MAT_s007
                                 + 0.976713 w_k1_MSFT_s007
                                 + 0.978662 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.23816 w_k1_ADM_s007
                                 + 1.11548 w_k1_AXP_s007
                                 + 1.57401 w_k1_DIS_s007
                                 + 1.09038 w_k1_FDX_s007
                                 + 1.59568 w_k1_IBM_s007
                                 + 1.1441 w_k1_INTC_s007 + 1.57175 w_k1_KO_s007
                                 + 1.60929 w_k1_MAT_s007
                                 + 1.13774 w_k1_MSFT_s007
                                 + 1.24082 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0779 w_k1_ADM_s007
                                 + 0.985614 w_k1_AXP_s007
                                 + 1.56653 w_k1_DIS_s007
                                 + 1.25412 w_k1_FDX_s007
                                 + 0.989269 w_k1_IBM_s007
                                 + 1.09902 w_k1_INTC_s007
                                 + 1.11812 w_k1_KO_s007 + 1.18932 w_k1_MAT_s007
                                 + 1.29586 w_k1_MSFT_s007
                                 + 1.11707 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.48248 w_k1_ADM_s010
                                 + 1.18393 w_k1_AXP_s010 + 1.5264 w_k1_DIS_s010
                                 + 1.29237 w_k1_FDX_s010
                                 + 1.11631 w_k1_IBM_s010
                                 + 1.10558 w_k1_INTC_s010
                                 + 1.29164 w_k1_KO_s010 + 1.08722 w_k1_MAT_s010
                                 + 1.1963 w_k1_MSFT_s010 + 1.28351 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.13525 w_k1_ADM_s010
                                 + 1.15783 w_k1_AXP_s010
                                 + 1.00408 w_k1_DIS_s010
                                 + 1.21791 w_k1_FDX_s010
                                 + 0.946301 w_k1_IBM_s010
                                 + 0.822939 w_k1_INTC_s010
                                 + 0.901175 w_k1_KO_s010
                                 + 0.923848 w_k1_MAT_s010
                                 + 0.96646 w_k1_MSFT_s010 + 1.0767 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.8618 w_k1_ADM_s010
                                 + 0.858192 w_k1_AXP_s010
                                 + 0.922119 w_k1_DIS_s010
                                 + 0.919787 w_k1_FDX_s010
                                 + 1.00016 w_k1_IBM_s010
                                 + 0.823126 w_k1_INTC_s010
                                 + 1.02693 w_k1_KO_s010
                                 + 0.871582 w_k1_MAT_s010
                                 + 0.879105 w_k1_MSFT_s010
                                 + 0.972827 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.86354 w_k1_ADM_s010
                                 + 0.817668 w_k1_AXP_s010
                                 + 0.700302 w_k1_DIS_s010
                                 + 0.807182 w_k1_FDX_s010
                                 + 0.716987 w_k1_IBM_s010
                                 + 0.722943 w_k1_INTC_s010
                                 + 0.659058 w_k1_KO_s010
                                 + 0.625293 w_k1_MAT_s010
                                 + 0.639481 w_k1_MSFT_s010
                                 + 0.769224 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.10346 w_k1_ADM_s010
                                 + 1.05851 w_k1_AXP_s010
                                 + 0.951142 w_k1_DIS_s010
                                 + 0.890819 w_k1_FDX_s010
                                 + 1.46647 w_k1_IBM_s010
                                 + 0.820231 w_k1_INTC_s010
                                 + 1.05465 w_k1_KO_s010 + 1.49253 w_k1_MAT_s010
                                 + 0.808418 w_k1_MSFT_s010
                                 + 0.96282 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.36187 w_k1_ADM_s010
                                 + 1.01171 w_k1_AXP_s010
                                 + 1.11274 w_k1_DIS_s010
                                 + 1.12524 w_k1_FDX_s010
                                 + 0.698022 w_k1_IBM_s010
                                 + 0.890245 w_k1_INTC_s010
                                 + 0.859866 w_k1_KO_s010
                                 + 0.778624 w_k1_MAT_s010
                                 + 0.990689 w_k1_MSFT_s010
                                 + 0.96768 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.851565 w_k1_ADM_s010
                                 + 0.796785 w_k1_AXP_s010
                                 + 0.74383 w_k1_DIS_s010
                                 + 0.773569 w_k1_FDX_s010
                                 + 1.14599 w_k1_IBM_s010
                                 + 0.771787 w_k1_INTC_s010
                                 + 0.865207 w_k1_KO_s010
                                 + 1.09211 w_k1_MAT_s010
                                 + 0.732099 w_k1_MSFT_s010
                                 + 0.977421 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.28734 w_k1_ADM_s010
                                 + 1.16518 w_k1_AXP_s010
                                 + 1.43109 w_k1_DIS_s010
                                 + 1.06617 w_k1_FDX_s010
                                 + 1.68777 w_k1_IBM_s010
                                 + 0.992603 w_k1_INTC_s010
                                 + 1.38606 w_k1_KO_s010 + 1.34763 w_k1_MAT_s010
                                 + 1.03929 w_k1_MSFT_s010
                                 + 1.35984 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.699949 w_k1_ADM_s011
                                 + 0.672321 w_k1_AXP_s011
                                 + 0.7501 w_k1_DIS_s011
                                 + 0.800552 w_k1_FDX_s011
                                 + 0.842715 w_k1_IBM_s011
                                 + 0.770971 w_k1_INTC_s011
                                 + 0.711325 w_k1_KO_s011
                                 + 0.803284 w_k1_MAT_s011
                                 + 0.782463 w_k1_MSFT_s011
                                 + 0.732514 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.931387 w_k1_ADM_s011
                                 + 0.861089 w_k1_AXP_s011
                                 + 1.01944 w_k1_DIS_s011
                                 + 0.920534 w_k1_FDX_s011
                                 + 0.948448 w_k1_IBM_s011
                                 + 0.697112 w_k1_INTC_s011
                                 + 0.8815 w_k1_KO_s011 + 0.852602 w_k1_MAT_s011
                                 + 0.903397 w_k1_MSFT_s011
                                 + 0.849379 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.928682 w_k1_ADM_s011
                                 + 1.43524 w_k1_AXP_s011
                                 + 1.76197 w_k1_DIS_s011
                                 + 1.27901 w_k1_FDX_s011
                                 + 1.43975 w_k1_IBM_s011
                                 + 0.954161 w_k1_INTC_s011
                                 + 1.28891 w_k1_KO_s011 + 1.68952 w_k1_MAT_s011
                                 + 1.18184 w_k1_MSFT_s011
                                 + 1.18925 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.969603 w_k1_ADM_s011
                                 + 0.603257 w_k1_AXP_s011
                                 + 0.884488 w_k1_DIS_s011
                                 + 0.97498 w_k1_FDX_s011
                                 + 0.615768 w_k1_IBM_s011
                                 + 0.718432 w_k1_INTC_s011
                                 + 0.742946 w_k1_KO_s011
                                 + 0.674425 w_k1_MAT_s011
                                 + 0.992476 w_k1_MSFT_s011
                                 + 0.788813 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.911813 w_k1_ADM_s011
                                 + 1.37242 w_k1_AXP_s011 + 1.2317 w_k1_DIS_s011
                                 + 0.946024 w_k1_FDX_s011
                                 + 1.6231 w_k1_IBM_s011
                                 + 1.01487 w_k1_INTC_s011
                                 + 1.28729 w_k1_KO_s011 + 1.48984 w_k1_MAT_s011
                                 + 1.01235 w_k1_MSFT_s011
                                 + 0.909373 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.933122 w_k1_ADM_s011
                                 + 0.967746 w_k1_AXP_s011
                                 + 1.14929 w_k1_DIS_s011
                                 + 1.01345 w_k1_FDX_s011
                                 + 1.15923 w_k1_IBM_s011
                                 + 0.856863 w_k1_INTC_s011
                                 + 1.06088 w_k1_KO_s011 + 1.02294 w_k1_MAT_s011
                                 + 1.00581 w_k1_MSFT_s011
                                 + 0.907442 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.23653 w_k1_ADM_s011
                                 + 0.881396 w_k1_AXP_s011
                                 + 1.14977 w_k1_DIS_s011
                                 + 1.10143 w_k1_FDX_s011
                                 + 0.632583 w_k1_IBM_s011
                                 + 0.764693 w_k1_INTC_s011
                                 + 0.791135 w_k1_KO_s011
                                 + 0.553088 w_k1_MAT_s011
                                 + 0.91889 w_k1_MSFT_s011
                                 + 0.81244 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.07432 w_k1_ADM_s011
                                 + 1.32755 w_k1_AXP_s011
                                 + 1.16398 w_k1_DIS_s011
                                 + 1.24287 w_k1_FDX_s011
                                 + 0.818826 w_k1_IBM_s011
                                 + 0.878068 w_k1_INTC_s011
                                 + 0.872458 w_k1_KO_s011
                                 + 0.881198 w_k1_MAT_s011
                                 + 0.936095 w_k1_MSFT_s011
                                 + 0.961601 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.00196 w_k1_ADM_s012
                                 + 1.22362 w_k1_AXP_s012
                                 + 0.892473 w_k1_DIS_s012
                                 + 0.886066 w_k1_FDX_s012
                                 + 1.44602 w_k1_IBM_s012
                                 + 1.06161 w_k1_INTC_s012
                                 + 1.26899 w_k1_KO_s012 + 1.29141 w_k1_MAT_s012
                                 + 0.921825 w_k1_MSFT_s012
                                 + 0.892438 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.02828 w_k1_ADM_s012
                                 + 0.918564 w_k1_AXP_s012
                                 + 0.944261 w_k1_DIS_s012
                                 + 1.09972 w_k1_FDX_s012
                                 + 0.85119 w_k1_IBM_s012
                                 + 0.995632 w_k1_INTC_s012
                                 + 0.94353 w_k1_KO_s012
                                 + 0.776545 w_k1_MAT_s012
                                 + 0.99491 w_k1_MSFT_s012
                                 + 0.962489 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.47681 w_k1_ADM_s012
                                 + 1.62563 w_k1_AXP_s012 + 1.4644 w_k1_DIS_s012
                                 + 1.36547 w_k1_FDX_s012 + 1.3301 w_k1_IBM_s012
                                 + 1.37844 w_k1_INTC_s012
                                 + 1.44096 w_k1_KO_s012 + 1.22849 w_k1_MAT_s012
                                 + 1.19159 w_k1_MSFT_s012 + 1.14 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.985497 w_k1_ADM_s012
                                 + 1.08055 w_k1_AXP_s012
                                 + 0.657914 w_k1_DIS_s012
                                 + 0.914694 w_k1_FDX_s012
                                 + 0.694835 w_k1_IBM_s012
                                 + 0.977139 w_k1_INTC_s012
                                 + 0.832156 w_k1_KO_s012
                                 + 0.803007 w_k1_MAT_s012
                                 + 0.801307 w_k1_MSFT_s012
                                 + 0.790879 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.884992 w_k1_ADM_s012
                                 + 0.793563 w_k1_AXP_s012
                                 + 0.743448 w_k1_DIS_s012
                                 + 0.75727 w_k1_FDX_s012
                                 + 0.951351 w_k1_IBM_s012
                                 + 0.785996 w_k1_INTC_s012
                                 + 1.05729 w_k1_KO_s012
                                 + 0.961723 w_k1_MAT_s012
                                 + 0.680389 w_k1_MSFT_s012
                                 + 0.834446 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02463 w_k1_ADM_s012
                                 + 1.0433 w_k1_AXP_s012
                                 + 0.800845 w_k1_DIS_s012
                                 + 0.892917 w_k1_FDX_s012
                                 + 0.880091 w_k1_IBM_s012
                                 + 0.943603 w_k1_INTC_s012
                                 + 0.845307 w_k1_KO_s012
                                 + 0.825065 w_k1_MAT_s012
                                 + 0.763102 w_k1_MSFT_s012
                                 + 0.899703 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.44505 w_k1_ADM_s012
                                 + 1.05081 w_k1_AXP_s012
                                 + 1.43865 w_k1_DIS_s012
                                 + 1.15039 w_k1_FDX_s012
                                 + 1.43744 w_k1_IBM_s012
                                 + 0.937513 w_k1_INTC_s012
                                 + 1.28592 w_k1_KO_s012 + 0.87202 w_k1_MAT_s012
                                 + 0.921704 w_k1_MSFT_s012
                                 + 1.08728 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.35816 w_k1_ADM_s012
                                 + 1.51597 w_k1_AXP_s012
                                 + 1.27664 w_k1_DIS_s012
                                 + 1.22768 w_k1_FDX_s012
                                 + 0.961386 w_k1_IBM_s012
                                 + 1.02438 w_k1_INTC_s012
                                 + 0.979632 w_k1_KO_s012
                                 + 1.14417 w_k1_MAT_s012
                                 + 1.01176 w_k1_MSFT_s012
                                 + 0.998763 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.843461 w_k1_ADM_s013
                                 + 0.693381 w_k1_AXP_s013
                                 + 0.865378 w_k1_DIS_s013
                                 + 0.74627 w_k1_FDX_s013
                                 + 0.938103 w_k1_IBM_s013
                                 + 0.838654 w_k1_INTC_s013
                                 + 0.809485 w_k1_KO_s013
                                 + 0.692258 w_k1_MAT_s013
                                 + 0.864064 w_k1_MSFT_s013
                                 + 0.799717 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.862694 w_k1_ADM_s013
                                 + 0.710294 w_k1_AXP_s013
                                 + 0.812897 w_k1_DIS_s013
                                 + 0.774088 w_k1_FDX_s013
                                 + 1.25863 w_k1_IBM_s013
                                 + 0.784588 w_k1_INTC_s013
                                 + 1.02253 w_k1_KO_s013 + 1.37608 w_k1_MAT_s013
                                 + 0.83049 w_k1_MSFT_s013
                                 + 0.976148 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.0819 w_k1_ADM_s013
                                 + 0.559305 w_k1_AXP_s013
                                 + 0.793618 w_k1_DIS_s013
                                 + 0.813463 w_k1_FDX_s013
                                 + 0.772706 w_k1_IBM_s013
                                 + 0.89804 w_k1_INTC_s013
                                 + 0.90439 w_k1_KO_s013
                                 + 0.588739 w_k1_MAT_s013
                                 + 0.983373 w_k1_MSFT_s013
                                 + 0.797946 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.16955 w_k1_ADM_s013
                                 + 0.84773 w_k1_AXP_s013
                                 + 0.944835 w_k1_DIS_s013
                                 + 0.960054 w_k1_FDX_s013
                                 + 0.771646 w_k1_IBM_s013
                                 + 0.782293 w_k1_INTC_s013
                                 + 0.824903 w_k1_KO_s013
                                 + 0.709704 w_k1_MAT_s013
                                 + 0.908962 w_k1_MSFT_s013
                                 + 0.928696 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.27642 w_k1_ADM_s013
                                 + 0.893973 w_k1_AXP_s013
                                 + 1.28666 w_k1_DIS_s013
                                 + 0.985953 w_k1_FDX_s013
                                 + 1.28557 w_k1_IBM_s013
                                 + 0.814518 w_k1_INTC_s013
                                 + 1.28208 w_k1_KO_s013 + 1.35051 w_k1_MAT_s013
                                 + 1.20425 w_k1_MSFT_s013
                                 + 1.06385 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.4851 w_k1_ADM_s013
                                 + 1.60446 w_k1_AXP_s013
                                 + 1.81522 w_k1_DIS_s013 + 1.3328 w_k1_FDX_s013
                                 + 1.28696 w_k1_IBM_s013
                                 + 1.21355 w_k1_INTC_s013
                                 + 1.21046 w_k1_KO_s013 + 1.51684 w_k1_MAT_s013
                                 + 1.3796 w_k1_MSFT_s013 + 1.31345 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.70047 w_k1_ADM_s013
                                 + 1.10833 w_k1_AXP_s013 + 1.396 w_k1_DIS_s013
                                 + 1.19756 w_k1_FDX_s013
                                 + 0.845928 w_k1_IBM_s013
                                 + 0.903537 w_k1_INTC_s013
                                 + 0.974139 w_k1_KO_s013
                                 + 0.830935 w_k1_MAT_s013
                                 + 1.21363 w_k1_MSFT_s013
                                 + 0.97824 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.06306 w_k1_ADM_s013
                                 + 0.90087 w_k1_AXP_s013
                                 + 0.931619 w_k1_DIS_s013
                                 + 0.85653 w_k1_FDX_s013
                                 + 1.02649 w_k1_IBM_s013
                                 + 0.845895 w_k1_INTC_s013
                                 + 0.964879 w_k1_KO_s013
                                 + 0.949993 w_k1_MAT_s013
                                 + 0.869402 w_k1_MSFT_s013
                                 + 0.940916 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.01605 w_k1_ADM_s014
                                 + 0.643549 w_k1_AXP_s014
                                 + 0.610586 w_k1_DIS_s014
                                 + 0.758068 w_k1_FDX_s014
                                 + 0.727826 w_k1_IBM_s014
                                 + 0.741902 w_k1_INTC_s014
                                 + 0.665932 w_k1_KO_s014
                                 + 0.794852 w_k1_MAT_s014
                                 + 0.653938 w_k1_MSFT_s014
                                 + 0.769073 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.77213 w_k1_ADM_s014
                                 + 1.29421 w_k1_AXP_s014
                                 + 1.73082 w_k1_DIS_s014
                                 + 1.32498 w_k1_FDX_s014
                                 + 1.63986 w_k1_IBM_s014
                                 + 1.22561 w_k1_INTC_s014
                                 + 1.30281 w_k1_KO_s014 + 1.51531 w_k1_MAT_s014
                                 + 1.23704 w_k1_MSFT_s014
                                 + 1.21524 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.25346 w_k1_ADM_s014
                                 + 1.17143 w_k1_AXP_s014
                                 + 0.888433 w_k1_DIS_s014
                                 + 0.993137 w_k1_FDX_s014
                                 + 1.14574 w_k1_IBM_s014
                                 + 0.960865 w_k1_INTC_s014
                                 + 1.11534 w_k1_KO_s014 + 1.28082 w_k1_MAT_s014
                                 + 0.85672 w_k1_MSFT_s014
                                 + 0.978714 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20244 w_k1_ADM_s014
                                 + 1.2759 w_k1_AXP_s014 + 1.39656 w_k1_DIS_s014
                                 + 1.20825 w_k1_FDX_s014
                                 + 1.04209 w_k1_IBM_s014
                                 + 1.16147 w_k1_INTC_s014
                                 + 1.15507 w_k1_KO_s014 + 1.39562 w_k1_MAT_s014
                                 + 1.2355 w_k1_MSFT_s014 + 1.16591 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.05639 w_k1_ADM_s014
                                 + 0.7211 w_k1_AXP_s014
                                 + 0.733962 w_k1_DIS_s014
                                 + 0.899754 w_k1_FDX_s014
                                 + 0.601064 w_k1_IBM_s014
                                 + 0.912662 w_k1_INTC_s014
                                 + 0.829548 w_k1_KO_s014
                                 + 0.678485 w_k1_MAT_s014
                                 + 0.834548 w_k1_MSFT_s014
                                 + 0.760779 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.950759 w_k1_ADM_s014
                                 + 0.622961 w_k1_AXP_s014
                                 + 0.930843 w_k1_DIS_s014
                                 + 0.87044 w_k1_FDX_s014
                                 + 0.982105 w_k1_IBM_s014
                                 + 1.21141 w_k1_INTC_s014
                                 + 1.02579 w_k1_KO_s014
                                 + 0.978596 w_k1_MAT_s014
                                 + 0.872259 w_k1_MSFT_s014
                                 + 0.849208 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.45239 w_k1_ADM_s014
                                 + 1.09624 w_k1_AXP_s014
                                 + 1.06292 w_k1_DIS_s014
                                 + 1.23543 w_k1_FDX_s014
                                 + 0.677569 w_k1_IBM_s014
                                 + 1.21486 w_k1_INTC_s014
                                 + 0.894459 w_k1_KO_s014
                                 + 0.632953 w_k1_MAT_s014
                                 + 0.962591 w_k1_MSFT_s014
                                 + 1.08935 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.909627 w_k1_ADM_s014
                                 + 0.856904 w_k1_AXP_s014
                                 + 0.677164 w_k1_DIS_s014
                                 + 0.835002 w_k1_FDX_s014
                                 + 1.08101 w_k1_IBM_s014
                                 + 1.02241 w_k1_INTC_s014
                                 + 0.766867 w_k1_KO_s014
                                 + 0.841318 w_k1_MAT_s014
                                 + 0.846098 w_k1_MSFT_s014
                                 + 0.78461 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.762834 w_k1_ADM_s015
                                 + 1.15015 w_k1_AXP_s015
                                 + 0.848751 w_k1_DIS_s015
                                 + 1.00866 w_k1_FDX_s015
                                 + 0.985652 w_k1_IBM_s015
                                 + 0.899338 w_k1_INTC_s015
                                 + 0.859794 w_k1_KO_s015
                                 + 1.19107 w_k1_MAT_s015
                                 + 0.759869 w_k1_MSFT_s015
                                 + 0.908647 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.791229 w_k1_ADM_s015
                                 + 0.845165 w_k1_AXP_s015
                                 + 0.603416 w_k1_DIS_s015
                                 + 0.843564 w_k1_FDX_s015
                                 + 0.805673 w_k1_IBM_s015
                                 + 0.987233 w_k1_INTC_s015
                                 + 0.702699 w_k1_KO_s015
                                 + 0.629171 w_k1_MAT_s015
                                 + 0.865522 w_k1_MSFT_s015
                                 + 0.801532 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.14784 w_k1_ADM_s015
                                 + 1.65494 w_k1_AXP_s015
                                 + 1.17171 w_k1_DIS_s015
                                 + 1.24051 w_k1_FDX_s015
                                 + 1.35986 w_k1_IBM_s015
                                 + 1.0036 w_k1_INTC_s015 + 1.15806 w_k1_KO_s015
                                 + 1.9033 w_k1_MAT_s015
                                 + 1.04693 w_k1_MSFT_s015
                                 + 1.04353 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.802324 w_k1_ADM_s015
                                 + 1.22144 w_k1_AXP_s015
                                 + 1.20294 w_k1_DIS_s015
                                 + 1.33405 w_k1_FDX_s015
                                 + 0.803699 w_k1_IBM_s015
                                 + 1.40743 w_k1_INTC_s015
                                 + 0.932482 w_k1_KO_s015
                                 + 0.726348 w_k1_MAT_s015
                                 + 1.13525 w_k1_MSFT_s015
                                 + 0.984383 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.861021 w_k1_ADM_s015
                                 + 0.90867 w_k1_AXP_s015
                                 + 1.29084 w_k1_DIS_s015
                                 + 1.15785 w_k1_FDX_s015
                                 + 1.21463 w_k1_IBM_s015
                                 + 1.11581 w_k1_INTC_s015
                                 + 1.17781 w_k1_KO_s015 + 1.01709 w_k1_MAT_s015
                                 + 1.08036 w_k1_MSFT_s015
                                 + 1.08188 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.806276 w_k1_ADM_s015
                                 + 0.909205 w_k1_AXP_s015
                                 + 0.77623 w_k1_DIS_s015
                                 + 0.878088 w_k1_FDX_s015
                                 + 1.14746 w_k1_IBM_s015
                                 + 0.803462 w_k1_INTC_s015
                                 + 0.837676 w_k1_KO_s015
                                 + 1.35215 w_k1_MAT_s015
                                 + 0.854094 w_k1_MSFT_s015
                                 + 0.798178 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.737692 w_k1_ADM_s015
                                 + 0.79112 w_k1_AXP_s015
                                 + 0.864669 w_k1_DIS_s015
                                 + 0.832374 w_k1_FDX_s015
                                 + 1.12803 w_k1_IBM_s015
                                 + 0.786212 w_k1_INTC_s015
                                 + 0.986065 w_k1_KO_s015 + 0.9956 w_k1_MAT_s015
                                 + 0.831902 w_k1_MSFT_s015
                                 + 0.90321 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.579877 w_k1_ADM_s015
                                 + 0.699583 w_k1_AXP_s015
                                 + 0.541983 w_k1_DIS_s015
                                 + 0.710146 w_k1_FDX_s015
                                 + 0.740869 w_k1_IBM_s015
                                 + 0.718442 w_k1_INTC_s015
                                 + 0.643458 w_k1_KO_s015
                                 + 0.628028 w_k1_MAT_s015
                                 + 0.58958 w_k1_MSFT_s015
                                 + 0.770053 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.23277 w_k1_ADM_s016
                                 + 1.56597 w_k1_AXP_s016
                                 + 1.82845 w_k1_DIS_s016
                                 + 1.43665 w_k1_FDX_s016
                                 + 1.12234 w_k1_IBM_s016
                                 + 1.07817 w_k1_INTC_s016
                                 + 1.25079 w_k1_KO_s016 + 1.20359 w_k1_MAT_s016
                                 + 1.25958 w_k1_MSFT_s016
                                 + 1.32449 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.969762 w_k1_ADM_s016
                                 + 0.913873 w_k1_AXP_s016
                                 + 1.12467 w_k1_DIS_s016
                                 + 0.969997 w_k1_FDX_s016
                                 + 0.972161 w_k1_IBM_s016
                                 + 0.942871 w_k1_INTC_s016
                                 + 1.0267 w_k1_KO_s016 + 0.75965 w_k1_MAT_s016
                                 + 0.991202 w_k1_MSFT_s016
                                 + 1.09246 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.787609 w_k1_ADM_s016
                                 + 0.879457 w_k1_AXP_s016
                                 + 0.824476 w_k1_DIS_s016
                                 + 0.767197 w_k1_FDX_s016
                                 + 1.23161 w_k1_IBM_s016
                                 + 0.828811 w_k1_INTC_s016
                                 + 0.820745 w_k1_KO_s016
                                 + 1.24424 w_k1_MAT_s016
                                 + 0.856604 w_k1_MSFT_s016
                                 + 0.996778 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.900168 w_k1_ADM_s016
                                 + 0.770002 w_k1_AXP_s016
                                 + 0.842281 w_k1_DIS_s016
                                 + 0.930818 w_k1_FDX_s016
                                 + 0.639508 w_k1_IBM_s016
                                 + 1.02292 w_k1_INTC_s016
                                 + 0.849878 w_k1_KO_s016
                                 + 0.576437 w_k1_MAT_s016
                                 + 0.955439 w_k1_MSFT_s016
                                 + 0.989998 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.05338 w_k1_ADM_s016
                                 + 1.11121 w_k1_AXP_s016 + 1.5888 w_k1_DIS_s016
                                 + 1.20949 w_k1_FDX_s016
                                 + 1.23354 w_k1_IBM_s016
                                 + 1.25933 w_k1_INTC_s016 + 1.5072 w_k1_KO_s016
                                 + 1.27096 w_k1_MAT_s016
                                 + 1.44643 w_k1_MSFT_s016
                                 + 1.19723 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.31798 w_k1_ADM_s016
                                 + 1.52149 w_k1_AXP_s016
                                 + 1.09855 w_k1_DIS_s016
                                 + 1.21751 w_k1_FDX_s016
                                 + 0.848942 w_k1_IBM_s016
                                 + 1.36171 w_k1_INTC_s016
                                 + 0.979177 w_k1_KO_s016
                                 + 0.826869 w_k1_MAT_s016
                                 + 1.01478 w_k1_MSFT_s016
                                 + 1.09819 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.804587 w_k1_ADM_s016
                                 + 0.674373 w_k1_AXP_s016
                                 + 0.624213 w_k1_DIS_s016
                                 + 0.73081 w_k1_FDX_s016
                                 + 0.814353 w_k1_IBM_s016
                                 + 0.752054 w_k1_INTC_s016
                                 + 0.755867 w_k1_KO_s016
                                 + 0.703005 w_k1_MAT_s016
                                 + 0.679527 w_k1_MSFT_s016
                                 + 0.884056 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.23246 w_k1_ADM_s016
                                 + 1.14302 w_k1_AXP_s016
                                 + 1.33204 w_k1_DIS_s016
                                 + 0.991293 w_k1_FDX_s016
                                 + 1.55207 w_k1_IBM_s016
                                 + 1.03526 w_k1_INTC_s016
                                 + 1.24121 w_k1_KO_s016 + 1.2784 w_k1_MAT_s016
                                 + 1.10441 w_k1_MSFT_s016
                                 + 1.32215 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.04645 w_k1_ADM_s017
                                 + 0.846142 w_k1_AXP_s017
                                 + 0.965071 w_k1_DIS_s017
                                 + 0.994043 w_k1_FDX_s017
                                 + 1.00791 w_k1_IBM_s017
                                 + 0.88409 w_k1_INTC_s017
                                 + 1.13413 w_k1_KO_s017 + 1.24004 w_k1_MAT_s017
                                 + 1.04371 w_k1_MSFT_s017
                                 + 0.990133 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.894057 w_k1_ADM_s017
                                 + 0.653514 w_k1_AXP_s017
                                 + 0.764331 w_k1_DIS_s017
                                 + 0.898049 w_k1_FDX_s017
                                 + 0.704393 w_k1_IBM_s017
                                 + 0.870455 w_k1_INTC_s017
                                 + 0.812874 w_k1_KO_s017
                                 + 0.605869 w_k1_MAT_s017
                                 + 0.926453 w_k1_MSFT_s017
                                 + 0.769701 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.30577 w_k1_ADM_s017
                                 + 1.35704 w_k1_AXP_s017
                                 + 1.30972 w_k1_DIS_s017
                                 + 1.33882 w_k1_FDX_s017
                                 + 1.08395 w_k1_IBM_s017
                                 + 1.23114 w_k1_INTC_s017
                                 + 1.28451 w_k1_KO_s017 + 1.74805 w_k1_MAT_s017
                                 + 1.26271 w_k1_MSFT_s017
                                 + 1.24935 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.819654 w_k1_ADM_s017
                                 + 1.10842 w_k1_AXP_s017
                                 + 0.857966 w_k1_DIS_s017
                                 + 0.909782 w_k1_FDX_s017
                                 + 1.27769 w_k1_IBM_s017
                                 + 1.03867 w_k1_INTC_s017
                                 + 1.10794 w_k1_KO_s017 + 1.37508 w_k1_MAT_s017
                                 + 0.969057 w_k1_MSFT_s017
                                 + 0.915969 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.923296 w_k1_ADM_s017
                                 + 1.21143 w_k1_AXP_s017
                                 + 1.25162 w_k1_DIS_s017
                                 + 1.17071 w_k1_FDX_s017
                                 + 1.73095 w_k1_IBM_s017
                                 + 1.26148 w_k1_INTC_s017
                                 + 1.22778 w_k1_KO_s017 + 1.65593 w_k1_MAT_s017
                                 + 0.936353 w_k1_MSFT_s017
                                 + 1.16699 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.13118 w_k1_ADM_s017
                                 + 0.902438 w_k1_AXP_s017
                                 + 1.24545 w_k1_DIS_s017
                                 + 1.20196 w_k1_FDX_s017
                                 + 1.09781 w_k1_IBM_s017
                                 + 1.33436 w_k1_INTC_s017
                                 + 1.24909 w_k1_KO_s017 + 1.01528 w_k1_MAT_s017
                                 + 1.16011 w_k1_MSFT_s017
                                 + 1.03912 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.792745 w_k1_ADM_s017
                                 + 0.77404 w_k1_AXP_s017
                                 + 0.687224 w_k1_DIS_s017
                                 + 0.865516 w_k1_FDX_s017
                                 + 0.908259 w_k1_IBM_s017
                                 + 0.989625 w_k1_INTC_s017
                                 + 0.742851 w_k1_KO_s017 + 1.149 w_k1_MAT_s017
                                 + 0.890395 w_k1_MSFT_s017
                                 + 0.800993 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.798763 w_k1_ADM_s017
                                 + 0.701366 w_k1_AXP_s017
                                 + 0.778765 w_k1_DIS_s017
                                 + 0.773753 w_k1_FDX_s017
                                 + 1.16873 w_k1_IBM_s017
                                 + 1.03509 w_k1_INTC_s017
                                 + 1.09916 w_k1_KO_s017 + 1.00784 w_k1_MAT_s017
                                 + 0.900164 w_k1_MSFT_s017
                                 + 0.953737 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.564207 w_k1_ADM_s100
                                 + 0.558479 w_k1_AXP_s100
                                 + 0.647085 w_k1_DIS_s100
                                 + 0.742175 w_k1_FDX_s100
                                 + 0.622764 w_k1_IBM_s100
                                 + 0.766702 w_k1_INTC_s100
                                 + 0.713425 w_k1_KO_s100
                                 + 0.60386 w_k1_MAT_s100
                                 + 0.814935 w_k1_MSFT_s100
                                 + 0.8653 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.06422 w_k1_ADM_s100
                                 + 1.13506 w_k1_AXP_s100
                                 + 1.11405 w_k1_DIS_s100
                                 + 1.18599 w_k1_FDX_s100
                                 + 0.897122 w_k1_IBM_s100
                                 + 1.15721 w_k1_INTC_s100
                                 + 1.00929 w_k1_KO_s100
                                 + 0.974632 w_k1_MAT_s100
                                 + 1.12996 w_k1_MSFT_s100
                                 + 1.08249 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.9194 w_k1_ADM_s100
                                 + 1.30467 w_k1_AXP_s100
                                 + 1.40862 w_k1_DIS_s100
                                 + 1.16481 w_k1_FDX_s100 + 1.3262 w_k1_IBM_s100
                                 + 1.22546 w_k1_INTC_s100
                                 + 1.28883 w_k1_KO_s100 + 1.42606 w_k1_MAT_s100
                                 + 1.25405 w_k1_MSFT_s100
                                 + 1.29519 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.967874 w_k1_ADM_s100
                                 + 1.19976 w_k1_AXP_s100
                                 + 0.935726 w_k1_DIS_s100
                                 + 0.989489 w_k1_FDX_s100
                                 + 1.16018 w_k1_IBM_s100
                                 + 1.17489 w_k1_INTC_s100
                                 + 1.02098 w_k1_KO_s100 + 1.2678 w_k1_MAT_s100
                                 + 0.913098 w_k1_MSFT_s100
                                 + 1.11951 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.752712 w_k1_ADM_s100
                                 + 0.762201 w_k1_AXP_s100
                                 + 0.796517 w_k1_DIS_s100
                                 + 0.840782 w_k1_FDX_s100
                                 + 0.846032 w_k1_IBM_s100
                                 + 0.94166 w_k1_INTC_s100
                                 + 0.742947 w_k1_KO_s100
                                 + 0.85332 w_k1_MAT_s100
                                 + 0.827782 w_k1_MSFT_s100
                                 + 0.852001 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.947202 w_k1_ADM_s100
                                 + 0.948622 w_k1_AXP_s100
                                 + 1.1557 w_k1_DIS_s100 + 1.17771 w_k1_FDX_s100
                                 + 0.686628 w_k1_IBM_s100
                                 + 1.26844 w_k1_INTC_s100
                                 + 0.884679 w_k1_KO_s100
                                 + 0.735239 w_k1_MAT_s100
                                 + 1.17377 w_k1_MSFT_s100
                                 + 1.19863 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.779162 w_k1_ADM_s100
                                 + 0.861974 w_k1_AXP_s100
                                 + 0.926833 w_k1_DIS_s100
                                 + 0.911253 w_k1_FDX_s100
                                 + 1.12676 w_k1_IBM_s100
                                 + 1.02243 w_k1_INTC_s100
                                 + 0.97976 w_k1_KO_s100 + 1.08604 w_k1_MAT_s100
                                 + 1.04398 w_k1_MSFT_s100
                                 + 1.05412 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.80592 w_k1_ADM_s100
                                 + 0.879587 w_k1_AXP_s100
                                 + 0.829661 w_k1_DIS_s100
                                 + 0.778637 w_k1_FDX_s100
                                 + 1.32533 w_k1_IBM_s100
                                 + 0.917357 w_k1_INTC_s100
                                 + 1.07444 w_k1_KO_s100 + 1.33784 w_k1_MAT_s100
                                 + 0.932816 w_k1_MSFT_s100
                                 + 0.999669 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.5237 w_k1_ADM_s101
                                 + 1.45961 w_k1_AXP_s101
                                 + 1.28458 w_k1_DIS_s101 + 1.201 w_k1_FDX_s101
                                 + 1.84273 w_k1_IBM_s101
                                 + 1.39072 w_k1_INTC_s101
                                 + 1.48462 w_k1_KO_s101 + 1.6414 w_k1_MAT_s101
                                 + 1.21421 w_k1_MSFT_s101
                                 + 1.05823 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.1441 w_k1_ADM_s101
                                 + 0.786781 w_k1_AXP_s101
                                 + 0.714135 w_k1_DIS_s101
                                 + 0.772406 w_k1_FDX_s101
                                 + 0.965877 w_k1_IBM_s101
                                 + 0.956223 w_k1_INTC_s101
                                 + 0.912003 w_k1_KO_s101
                                 + 0.800973 w_k1_MAT_s101
                                 + 0.797748 w_k1_MSFT_s101
                                 + 0.937564 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.4809 w_k1_ADM_s101
                                 + 1.80217 w_k1_AXP_s101
                                 + 1.29257 w_k1_DIS_s101
                                 + 1.33863 w_k1_FDX_s101
                                 + 1.30587 w_k1_IBM_s101
                                 + 1.28664 w_k1_INTC_s101
                                 + 1.25899 w_k1_KO_s101 + 1.53879 w_k1_MAT_s101
                                 + 1.04693 w_k1_MSFT_s101
                                 + 1.15906 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.50117 w_k1_ADM_s101
                                 + 1.1727 w_k1_AXP_s101 + 1.33371 w_k1_DIS_s101
                                 + 1.39826 w_k1_FDX_s101
                                 + 1.00162 w_k1_IBM_s101
                                 + 1.48359 w_k1_INTC_s101 + 1.1685 w_k1_KO_s101
                                 + 0.725287 w_k1_MAT_s101
                                 + 1.40679 w_k1_MSFT_s101
                                 + 1.13727 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.3902 w_k1_ADM_s101
                                 + 1.05223 w_k1_AXP_s101
                                 + 1.05271 w_k1_DIS_s101 + 1.1785 w_k1_FDX_s101
                                 + 0.782465 w_k1_IBM_s101
                                 + 1.37242 w_k1_INTC_s101
                                 + 0.933337 w_k1_KO_s101
                                 + 0.703405 w_k1_MAT_s101
                                 + 1.14909 w_k1_MSFT_s101
                                 + 1.04628 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.29339 w_k1_ADM_s101
                                 + 1.12148 w_k1_AXP_s101
                                 + 1.11743 w_k1_DIS_s101
                                 + 1.07796 w_k1_FDX_s101
                                 + 1.26501 w_k1_IBM_s101
                                 + 1.12519 w_k1_INTC_s101
                                 + 1.36189 w_k1_KO_s101 + 1.25876 w_k1_MAT_s101
                                 + 1.17836 w_k1_MSFT_s101
                                 + 1.02636 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.07801 w_k1_ADM_s101
                                 + 1.25833 w_k1_AXP_s101
                                 + 1.16558 w_k1_DIS_s101
                                 + 0.953412 w_k1_FDX_s101
                                 + 1.3924 w_k1_IBM_s101
                                 + 1.20569 w_k1_INTC_s101
                                 + 1.54566 w_k1_KO_s101 + 1.49012 w_k1_MAT_s101
                                 + 1.10532 w_k1_MSFT_s101 + 1.029 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.946564 w_k1_ADM_s101
                                 + 0.806042 w_k1_AXP_s101
                                 + 0.812505 w_k1_DIS_s101
                                 + 0.95034 w_k1_FDX_s101
                                 + 0.742348 w_k1_IBM_s101
                                 + 1.05236 w_k1_INTC_s101
                                 + 0.844593 w_k1_KO_s101
                                 + 0.590318 w_k1_MAT_s101
                                 + 0.966137 w_k1_MSFT_s101
                                 + 0.840791 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.85257 w_k1_ADM_s102
                                 + 1.21836 w_k1_AXP_s102
                                 + 1.07683 w_k1_DIS_s102
                                 + 0.998166 w_k1_FDX_s102
                                 + 1.6506 w_k1_IBM_s102 + 1.0579 w_k1_INTC_s102
                                 + 1.21353 w_k1_KO_s102 + 2.11322 w_k1_MAT_s102
                                 + 0.877345 w_k1_MSFT_s102
                                 + 1.0636 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.774346 w_k1_ADM_s102
                                 + 1.14175 w_k1_AXP_s102
                                 + 0.838684 w_k1_DIS_s102
                                 + 0.875883 w_k1_FDX_s102
                                 + 1.26094 w_k1_IBM_s102
                                 + 0.90659 w_k1_INTC_s102
                                 + 1.00884 w_k1_KO_s102 + 1.63931 w_k1_MAT_s102
                                 + 0.768869 w_k1_MSFT_s102
                                 + 1.09391 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.12002 w_k1_ADM_s102
                                 + 1.49112 w_k1_AXP_s102
                                 + 1.64132 w_k1_DIS_s102
                                 + 1.22038 w_k1_FDX_s102 + 1.6175 w_k1_IBM_s102
                                 + 1.02668 w_k1_INTC_s102
                                 + 1.39926 w_k1_KO_s102 + 2.65162 w_k1_MAT_s102
                                 + 1.19763 w_k1_MSFT_s102
                                 + 1.19919 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.92062 w_k1_ADM_s102
                                 + 0.732553 w_k1_AXP_s102
                                 + 0.80114 w_k1_DIS_s102
                                 + 0.804832 w_k1_FDX_s102
                                 + 0.85009 w_k1_IBM_s102
                                 + 0.819706 w_k1_INTC_s102
                                 + 0.846957 w_k1_KO_s102
                                 + 0.929908 w_k1_MAT_s102
                                 + 0.838594 w_k1_MSFT_s102
                                 + 0.974048 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.10153 w_k1_ADM_s102
                                 + 1.02108 w_k1_AXP_s102
                                 + 1.05029 w_k1_DIS_s102
                                 + 1.09725 w_k1_FDX_s102
                                 + 0.736862 w_k1_IBM_s102
                                 + 0.996855 w_k1_INTC_s102
                                 + 0.866918 w_k1_KO_s102
                                 + 1.17289 w_k1_MAT_s102
                                 + 1.0884 w_k1_MSFT_s102 + 1.09665 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.816481 w_k1_ADM_s102
                                 + 1.27757 w_k1_AXP_s102
                                 + 1.24442 w_k1_DIS_s102
                                 + 1.16607 w_k1_FDX_s102 + 1.0334 w_k1_IBM_s102
                                 + 1.02032 w_k1_INTC_s102
                                 + 1.04165 w_k1_KO_s102 + 1.5526 w_k1_MAT_s102
                                 + 1.10735 w_k1_MSFT_s102
                                 + 1.11215 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.07125 w_k1_ADM_s102
                                 + 0.970392 w_k1_AXP_s102
                                 + 1.18429 w_k1_DIS_s102
                                 + 0.96951 w_k1_FDX_s102
                                 + 1.26735 w_k1_IBM_s102
                                 + 0.893664 w_k1_INTC_s102
                                 + 1.20767 w_k1_KO_s102 + 2.01756 w_k1_MAT_s102
                                 + 1.00171 w_k1_MSFT_s102
                                 + 1.24898 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.30318 w_k1_ADM_s102
                                 + 1.51258 w_k1_AXP_s102
                                 + 1.55002 w_k1_DIS_s102
                                 + 1.46734 w_k1_FDX_s102
                                 + 1.06253 w_k1_IBM_s102
                                 + 1.02475 w_k1_INTC_s102
                                 + 1.16156 w_k1_KO_s102 + 1.66407 w_k1_MAT_s102
                                 + 1.18278 w_k1_MSFT_s102
                                 + 1.30463 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.25488 w_k1_ADM_s103
                                 + 1.07532 w_k1_AXP_s103
                                 + 1.40143 w_k1_DIS_s103
                                 + 1.24492 w_k1_FDX_s103
                                 + 0.968285 w_k1_IBM_s103
                                 + 1.16419 w_k1_INTC_s103
                                 + 1.02255 w_k1_KO_s103 + 0.96228 w_k1_MAT_s103
                                 + 1.21163 w_k1_MSFT_s103
                                 + 1.18685 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.977067 w_k1_ADM_s103
                                 + 0.972773 w_k1_AXP_s103
                                 + 0.747297 w_k1_DIS_s103
                                 + 0.962999 w_k1_FDX_s103
                                 + 0.83144 w_k1_IBM_s103
                                 + 0.963437 w_k1_INTC_s103
                                 + 0.900116 w_k1_KO_s103
                                 + 0.870963 w_k1_MAT_s103
                                 + 0.859949 w_k1_MSFT_s103
                                 + 1.0009 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.02022 w_k1_ADM_s103
                                 + 1.21285 w_k1_AXP_s103
                                 + 1.30117 w_k1_DIS_s103
                                 + 1.11703 w_k1_FDX_s103
                                 + 1.53814 w_k1_IBM_s103
                                 + 1.05124 w_k1_INTC_s103
                                 + 1.17471 w_k1_KO_s103 + 1.55754 w_k1_MAT_s103
                                 + 1.05316 w_k1_MSFT_s103
                                 + 1.04342 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.51077 w_k1_ADM_s103
                                 + 1.76084 w_k1_AXP_s103
                                 + 1.54277 w_k1_DIS_s103
                                 + 1.62999 w_k1_FDX_s103
                                 + 1.14342 w_k1_IBM_s103
                                 + 1.24917 w_k1_INTC_s103
                                 + 1.11125 w_k1_KO_s103 + 1.40699 w_k1_MAT_s103
                                 + 1.40214 w_k1_MSFT_s103
                                 + 1.31693 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.17231 w_k1_ADM_s103
                                 + 1.19707 w_k1_AXP_s103
                                 + 0.974523 w_k1_DIS_s103
                                 + 1.22157 w_k1_FDX_s103
                                 + 0.940009 w_k1_IBM_s103
                                 + 1.1927 w_k1_INTC_s103 + 1.09851 w_k1_KO_s103
                                 + 1.26202 w_k1_MAT_s103
                                 + 1.1775 w_k1_MSFT_s103 + 1.09457 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.895945 w_k1_ADM_s103
                                 + 0.75788 w_k1_AXP_s103
                                 + 1.02897 w_k1_DIS_s103
                                 + 0.962951 w_k1_FDX_s103
                                 + 0.8421 w_k1_IBM_s103
                                 + 0.995138 w_k1_INTC_s103
                                 + 0.954138 w_k1_KO_s103
                                 + 0.61637 w_k1_MAT_s103
                                 + 0.943431 w_k1_MSFT_s103
                                 + 0.966124 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.758885 w_k1_ADM_s103
                                 + 0.59407 w_k1_AXP_s103
                                 + 0.543069 w_k1_DIS_s103
                                 + 0.721367 w_k1_FDX_s103
                                 + 0.765404 w_k1_IBM_s103
                                 + 0.828993 w_k1_INTC_s103
                                 + 0.666599 w_k1_KO_s103
                                 + 0.664319 w_k1_MAT_s103
                                 + 0.717751 w_k1_MSFT_s103
                                 + 0.889843 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.925549 w_k1_ADM_s103
                                 + 0.959091 w_k1_AXP_s103
                                 + 0.838689 w_k1_DIS_s103
                                 + 0.884488 w_k1_FDX_s103
                                 + 1.22766 w_k1_IBM_s103
                                 + 0.836244 w_k1_INTC_s103
                                 + 0.951838 w_k1_KO_s103
                                 + 1.19007 w_k1_MAT_s103
                                 + 0.855909 w_k1_MSFT_s103
                                 + 1.00523 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.00872 w_k1_ADM_s104
                                 + 1.59804 w_k1_AXP_s104
                                 + 1.75799 w_k1_DIS_s104 + 1.3728 w_k1_FDX_s104
                                 + 1.23176 w_k1_IBM_s104
                                 + 1.43707 w_k1_INTC_s104
                                 + 1.62218 w_k1_KO_s104 + 1.83786 w_k1_MAT_s104
                                 + 1.30787 w_k1_MSFT_s104
                                 + 1.37684 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.886715 w_k1_ADM_s104
                                 + 1.18566 w_k1_AXP_s104
                                 + 0.971034 w_k1_DIS_s104
                                 + 1.11867 w_k1_FDX_s104
                                 + 0.761864 w_k1_IBM_s104
                                 + 1.11462 w_k1_INTC_s104
                                 + 0.854262 w_k1_KO_s104
                                 + 0.721548 w_k1_MAT_s104
                                 + 1.12293 w_k1_MSFT_s104
                                 + 1.07985 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.817751 w_k1_ADM_s104
                                 + 0.806279 w_k1_AXP_s104
                                 + 0.687289 w_k1_DIS_s104
                                 + 0.705746 w_k1_FDX_s104
                                 + 1.09281 w_k1_IBM_s104
                                 + 0.795398 w_k1_INTC_s104
                                 + 0.928899 w_k1_KO_s104
                                 + 1.09944 w_k1_MAT_s104
                                 + 0.768176 w_k1_MSFT_s104
                                 + 0.92637 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.03424 w_k1_ADM_s104
                                 + 1.09544 w_k1_AXP_s104
                                 + 1.10272 w_k1_DIS_s104
                                 + 1.06375 w_k1_FDX_s104
                                 + 1.05821 w_k1_IBM_s104
                                 + 0.986412 w_k1_INTC_s104
                                 + 1.24236 w_k1_KO_s104 + 1.22142 w_k1_MAT_s104
                                 + 1.06305 w_k1_MSFT_s104
                                 + 1.13806 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.36389 w_k1_ADM_s104
                                 + 1.60636 w_k1_AXP_s104
                                 + 1.14567 w_k1_DIS_s104
                                 + 1.17077 w_k1_FDX_s104
                                 + 1.39414 w_k1_IBM_s104
                                 + 0.985525 w_k1_INTC_s104
                                 + 1.43729 w_k1_KO_s104 + 1.57289 w_k1_MAT_s104
                                 + 1.14057 w_k1_MSFT_s104
                                 + 1.32144 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.79828 w_k1_ADM_s104
                                 + 0.817522 w_k1_AXP_s104
                                 + 0.7681 w_k1_DIS_s104
                                 + 0.821537 w_k1_FDX_s104
                                 + 0.930247 w_k1_IBM_s104
                                 + 0.93698 w_k1_INTC_s104
                                 + 1.00101 w_k1_KO_s104
                                 + 0.904131 w_k1_MAT_s104
                                 + 0.880734 w_k1_MSFT_s104
                                 + 0.88272 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.820066 w_k1_ADM_s104
                                 + 0.786185 w_k1_AXP_s104
                                 + 0.599441 w_k1_DIS_s104
                                 + 0.784633 w_k1_FDX_s104
                                 + 0.613539 w_k1_IBM_s104
                                 + 0.959773 w_k1_INTC_s104
                                 + 0.654048 w_k1_KO_s104
                                 + 0.576373 w_k1_MAT_s104
                                 + 0.813951 w_k1_MSFT_s104
                                 + 0.859505 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.87323 w_k1_ADM_s104
                                 + 0.741469 w_k1_AXP_s104
                                 + 0.894227 w_k1_DIS_s104
                                 + 0.88085 w_k1_FDX_s104
                                 + 0.783273 w_k1_IBM_s104
                                 + 0.938843 w_k1_INTC_s104
                                 + 0.925909 w_k1_KO_s104
                                 + 0.721476 w_k1_MAT_s104
                                 + 0.971213 w_k1_MSFT_s104
                                 + 1.03852 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.97746 w_k1_ADM_s105
                                 + 1.09274 w_k1_AXP_s105
                                 + 1.05399 w_k1_DIS_s105
                                 + 0.886633 w_k1_FDX_s105
                                 + 1.48589 w_k1_IBM_s105
                                 + 0.918231 w_k1_INTC_s105
                                 + 1.17224 w_k1_KO_s105 + 1.70168 w_k1_MAT_s105
                                 + 0.917244 w_k1_MSFT_s105
                                 + 1.03403 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.18465 w_k1_ADM_s105
                                 + 1.36075 w_k1_AXP_s105
                                 + 1.01922 w_k1_DIS_s105 + 1.0456 w_k1_FDX_s105
                                 + 1.06051 w_k1_IBM_s105
                                 + 0.914916 w_k1_INTC_s105
                                 + 0.930143 w_k1_KO_s105
                                 + 1.47192 w_k1_MAT_s105
                                 + 0.846869 w_k1_MSFT_s105
                                 + 1.07632 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.09622 w_k1_ADM_s105
                                 + 0.70136 w_k1_AXP_s105
                                 + 0.683003 w_k1_DIS_s105
                                 + 0.730877 w_k1_FDX_s105
                                 + 0.640149 w_k1_IBM_s105
                                 + 0.588319 w_k1_INTC_s105
                                 + 0.669766 w_k1_KO_s105
                                 + 0.624794 w_k1_MAT_s105
                                 + 0.665817 w_k1_MSFT_s105
                                 + 0.894485 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.71716 w_k1_ADM_s105
                                 + 1.19189 w_k1_AXP_s105
                                 + 1.51398 w_k1_DIS_s105
                                 + 1.19546 w_k1_FDX_s105
                                 + 1.24323 w_k1_IBM_s105
                                 + 1.05451 w_k1_INTC_s105
                                 + 1.28326 w_k1_KO_s105 + 1.22259 w_k1_MAT_s105
                                 + 1.0781 w_k1_MSFT_s105 + 1.16744 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.92151 w_k1_ADM_s105
                                 + 0.92304 w_k1_AXP_s105
                                 + 0.879189 w_k1_DIS_s105
                                 + 0.829225 w_k1_FDX_s105
                                 + 0.886371 w_k1_IBM_s105
                                 + 0.939985 w_k1_INTC_s105
                                 + 0.894625 w_k1_KO_s105
                                 + 1.16565 w_k1_MAT_s105
                                 + 0.869851 w_k1_MSFT_s105
                                 + 0.89423 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.13655 w_k1_ADM_s105
                                 + 1.07546 w_k1_AXP_s105
                                 + 1.16024 w_k1_DIS_s105
                                 + 1.19416 w_k1_FDX_s105
                                 + 0.891342 w_k1_IBM_s105
                                 + 0.887217 w_k1_INTC_s105
                                 + 1.03001 w_k1_KO_s105 + 1.31895 w_k1_MAT_s105
                                 + 0.995695 w_k1_MSFT_s105
                                 + 1.00898 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.864376 w_k1_ADM_s105
                                 + 0.973202 w_k1_AXP_s105
                                 + 1.16577 w_k1_DIS_s105
                                 + 1.08029 w_k1_FDX_s105
                                 + 0.771778 w_k1_IBM_s105
                                 + 0.933688 w_k1_INTC_s105
                                 + 0.943509 w_k1_KO_s105
                                 + 0.866741 w_k1_MAT_s105
                                 + 1.01281 w_k1_MSFT_s105
                                 + 0.938702 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.721089 w_k1_ADM_s105
                                 + 0.71343 w_k1_AXP_s105
                                 + 0.641134 w_k1_DIS_s105
                                 + 0.715344 w_k1_FDX_s105
                                 + 1.11334 w_k1_IBM_s105
                                 + 0.720959 w_k1_INTC_s105
                                 + 0.885901 w_k1_KO_s105
                                 + 1.23374 w_k1_MAT_s105
                                 + 0.720348 w_k1_MSFT_s105
                                 + 0.9176 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.35792 w_k1_ADM_s106
                                 + 1.29805 w_k1_AXP_s106
                                 + 1.32899 w_k1_DIS_s106
                                 + 1.11711 w_k1_FDX_s106
                                 + 1.45172 w_k1_IBM_s106
                                 + 1.12788 w_k1_INTC_s106
                                 + 1.26538 w_k1_KO_s106 + 1.60202 w_k1_MAT_s106
                                 + 1.07499 w_k1_MSFT_s106
                                 + 1.03948 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.889455 w_k1_ADM_s106
                                 + 0.910628 w_k1_AXP_s106
                                 + 0.904556 w_k1_DIS_s106
                                 + 0.935449 w_k1_FDX_s106
                                 + 0.800569 w_k1_IBM_s106
                                 + 0.993958 w_k1_INTC_s106
                                 + 0.86746 w_k1_KO_s106 + 1.05306 w_k1_MAT_s106
                                 + 0.95593 w_k1_MSFT_s106
                                 + 0.972792 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.967358 w_k1_ADM_s106
                                 + 1.43766 w_k1_AXP_s106
                                 + 1.17797 w_k1_DIS_s106 + 1.1416 w_k1_FDX_s106
                                 + 0.92997 w_k1_IBM_s106
                                 + 0.956042 w_k1_INTC_s106
                                 + 0.906653 w_k1_KO_s106
                                 + 1.45703 w_k1_MAT_s106
                                 + 1.06596 w_k1_MSFT_s106
                                 + 1.04126 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.14923 w_k1_ADM_s106
                                 + 0.981051 w_k1_AXP_s106
                                 + 0.790853 w_k1_DIS_s106
                                 + 1.00172 w_k1_FDX_s106
                                 + 0.603255 w_k1_IBM_s106
                                 + 0.877005 w_k1_INTC_s106
                                 + 0.695515 w_k1_KO_s106
                                 + 0.579972 w_k1_MAT_s106
                                 + 0.851832 w_k1_MSFT_s106
                                 + 0.896046 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.25465 w_k1_ADM_s106
                                 + 1.59976 w_k1_AXP_s106
                                 + 1.47077 w_k1_DIS_s106
                                 + 1.43953 w_k1_FDX_s106
                                 + 1.09665 w_k1_IBM_s106
                                 + 1.15952 w_k1_INTC_s106
                                 + 1.16623 w_k1_KO_s106 + 1.48077 w_k1_MAT_s106
                                 + 1.28161 w_k1_MSFT_s106
                                 + 1.36331 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.991085 w_k1_ADM_s106
                                 + 1.05705 w_k1_AXP_s106
                                 + 0.631845 w_k1_DIS_s106
                                 + 0.74993 w_k1_FDX_s106
                                 + 0.786249 w_k1_IBM_s106
                                 + 0.787311 w_k1_INTC_s106
                                 + 0.71261 w_k1_KO_s106
                                 + 0.840317 w_k1_MAT_s106
                                 + 0.759603 w_k1_MSFT_s106
                                 + 0.849286 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.04872 w_k1_ADM_s106
                                 + 1.00064 w_k1_AXP_s106
                                 + 0.846212 w_k1_DIS_s106
                                 + 0.892618 w_k1_FDX_s106
                                 + 1.41984 w_k1_IBM_s106
                                 + 0.96161 w_k1_INTC_s106
                                 + 1.08856 w_k1_KO_s106 + 1.30506 w_k1_MAT_s106
                                 + 0.907156 w_k1_MSFT_s106
                                 + 1.07345 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.981006 w_k1_ADM_s106
                                 + 0.978468 w_k1_AXP_s106
                                 + 0.823238 w_k1_DIS_s106
                                 + 0.872772 w_k1_FDX_s106
                                 + 1.10852 w_k1_IBM_s106
                                 + 1.08765 w_k1_INTC_s106
                                 + 1.03213 w_k1_KO_s106
                                 + 0.885566 w_k1_MAT_s106
                                 + 1.01409 w_k1_MSFT_s106
                                 + 0.973539 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.877314 w_k1_ADM_s107
                                 + 1.01999 w_k1_AXP_s107
                                 + 1.03984 w_k1_DIS_s107
                                 + 0.916466 w_k1_FDX_s107
                                 + 1.18439 w_k1_IBM_s107
                                 + 0.916445 w_k1_INTC_s107
                                 + 0.918916 w_k1_KO_s107
                                 + 1.23903 w_k1_MAT_s107
                                 + 0.871074 w_k1_MSFT_s107
                                 + 0.918115 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.00189 w_k1_ADM_s107
                                 + 1.29946 w_k1_AXP_s107
                                 + 1.10181 w_k1_DIS_s107
                                 + 1.05548 w_k1_FDX_s107
                                 + 1.46645 w_k1_IBM_s107
                                 + 1.06929 w_k1_INTC_s107
                                 + 1.11565 w_k1_KO_s107 + 1.76249 w_k1_MAT_s107
                                 + 0.940576 w_k1_MSFT_s107 + 1.063 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.921584 w_k1_ADM_s107
                                 + 0.817564 w_k1_AXP_s107
                                 + 0.723206 w_k1_DIS_s107
                                 + 0.815011 w_k1_FDX_s107
                                 + 0.735268 w_k1_IBM_s107
                                 + 0.828104 w_k1_INTC_s107
                                 + 0.739375 w_k1_KO_s107
                                 + 0.640528 w_k1_MAT_s107
                                 + 0.746481 w_k1_MSFT_s107
                                 + 0.842516 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.20314 w_k1_ADM_s107
                                 + 1.38439 w_k1_AXP_s107
                                 + 1.23448 w_k1_DIS_s107
                                 + 1.24012 w_k1_FDX_s107
                                 + 0.800615 w_k1_IBM_s107
                                 + 1.16574 w_k1_INTC_s107
                                 + 0.80952 w_k1_KO_s107
                                 + 0.856253 w_k1_MAT_s107
                                 + 1.13221 w_k1_MSFT_s107
                                 + 1.00389 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.25591 w_k1_ADM_s107
                                 + 1.30138 w_k1_AXP_s107
                                 + 1.53178 w_k1_DIS_s107
                                 + 1.03525 w_k1_FDX_s107
                                 + 1.93664 w_k1_IBM_s107
                                 + 0.950656 w_k1_INTC_s107
                                 + 1.34124 w_k1_KO_s107 + 2.02233 w_k1_MAT_s107
                                 + 1.15027 w_k1_MSFT_s107
                                 + 1.26711 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.71956 w_k1_ADM_s107
                                 + 0.753319 w_k1_AXP_s107
                                 + 0.6528 w_k1_DIS_s107
                                 + 0.640624 w_k1_FDX_s107
                                 + 1.21733 w_k1_IBM_s107
                                 + 0.725285 w_k1_INTC_s107
                                 + 0.710555 w_k1_KO_s107
                                 + 1.15847 w_k1_MAT_s107
                                 + 0.719615 w_k1_MSFT_s107
                                 + 0.841245 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.890507 w_k1_ADM_s107
                                 + 0.882353 w_k1_AXP_s107
                                 + 1.11874 w_k1_DIS_s107
                                 + 0.940981 w_k1_FDX_s107
                                 + 0.983364 w_k1_IBM_s107
                                 + 1.18516 w_k1_INTC_s107
                                 + 1.05088 w_k1_KO_s107
                                 + 0.972083 w_k1_MAT_s107
                                 + 1.05908 w_k1_MSFT_s107 + 1.1161 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.05432 w_k1_ADM_s107
                                 + 1.46328 w_k1_AXP_s107
                                 + 1.63501 w_k1_DIS_s107
                                 + 1.44541 w_k1_FDX_s107
                                 + 1.09229 w_k1_IBM_s107
                                 + 1.52548 w_k1_INTC_s107
                                 + 1.08999 w_k1_KO_s107 + 1.20876 w_k1_MAT_s107
                                 + 1.41553 w_k1_MSFT_s107
                                 + 1.15379 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.904875 w_k1_ADM_s110
                                 + 0.928179 w_k1_AXP_s110
                                 + 0.979157 w_k1_DIS_s110
                                 + 1.01428 w_k1_FDX_s110
                                 + 1.17783 w_k1_IBM_s110
                                 + 0.935831 w_k1_INTC_s110
                                 + 0.966541 w_k1_KO_s110
                                 + 1.00142 w_k1_MAT_s110
                                 + 0.899379 w_k1_MSFT_s110
                                 + 1.0216 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.15557 w_k1_ADM_s110
                                 + 1.50669 w_k1_AXP_s110
                                 + 1.10431 w_k1_DIS_s110 + 1.0448 w_k1_FDX_s110
                                 + 1.55345 w_k1_IBM_s110
                                 + 1.11242 w_k1_INTC_s110
                                 + 1.04691 w_k1_KO_s110 + 2.00851 w_k1_MAT_s110
                                 + 0.944591 w_k1_MSFT_s110
                                 + 1.03672 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.967279 w_k1_ADM_s110
                                 + 0.969131 w_k1_AXP_s110
                                 + 0.788134 w_k1_DIS_s110
                                 + 0.834742 w_k1_FDX_s110
                                 + 1.17833 w_k1_IBM_s110
                                 + 0.992986 w_k1_INTC_s110
                                 + 0.892318 w_k1_KO_s110
                                 + 1.02604 w_k1_MAT_s110
                                 + 0.752786 w_k1_MSFT_s110
                                 + 0.887623 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.16464 w_k1_ADM_s110
                                 + 1.22785 w_k1_AXP_s110
                                 + 1.22099 w_k1_DIS_s110
                                 + 1.14663 w_k1_FDX_s110
                                 + 1.05565 w_k1_IBM_s110
                                 + 1.14819 w_k1_INTC_s110
                                 + 1.06973 w_k1_KO_s110
                                 + 0.878337 w_k1_MAT_s110
                                 + 1.12522 w_k1_MSFT_s110
                                 + 1.02364 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.698076 w_k1_ADM_s110
                                 + 0.733728 w_k1_AXP_s110
                                 + 0.690092 w_k1_DIS_s110
                                 + 0.787265 w_k1_FDX_s110
                                 + 0.806418 w_k1_IBM_s110
                                 + 0.887711 w_k1_INTC_s110
                                 + 0.67379 w_k1_KO_s110
                                 + 0.868736 w_k1_MAT_s110
                                 + 0.862779 w_k1_MSFT_s110
                                 + 0.858038 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06336 w_k1_ADM_s110
                                 + 0.910585 w_k1_AXP_s110
                                 + 0.929186 w_k1_DIS_s110
                                 + 0.998771 w_k1_FDX_s110
                                 + 0.760086 w_k1_IBM_s110
                                 + 0.956022 w_k1_INTC_s110
                                 + 0.769882 w_k1_KO_s110
                                 + 0.777714 w_k1_MAT_s110
                                 + 1.01755 w_k1_MSFT_s110
                                 + 0.962797 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.38946 w_k1_ADM_s110
                                 + 1.32956 w_k1_AXP_s110
                                 + 1.54491 w_k1_DIS_s110 + 1.3822 w_k1_FDX_s110
                                 + 1.06244 w_k1_IBM_s110
                                 + 1.39744 w_k1_INTC_s110
                                 + 1.16401 w_k1_KO_s110
                                 + 0.774995 w_k1_MAT_s110
                                 + 1.42715 w_k1_MSFT_s110
                                 + 1.19353 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.927723 w_k1_ADM_s110
                                 + 1.56403 w_k1_AXP_s110
                                 + 1.50046 w_k1_DIS_s110
                                 + 1.29853 w_k1_FDX_s110
                                 + 1.58449 w_k1_IBM_s110
                                 + 1.43139 w_k1_INTC_s110
                                 + 1.30937 w_k1_KO_s110 + 1.92793 w_k1_MAT_s110
                                 + 1.26909 w_k1_MSFT_s110
                                 + 1.26165 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.78943 w_k1_ADM_s111
                                 + 0.669007 w_k1_AXP_s111
                                 + 0.818311 w_k1_DIS_s111
                                 + 0.837775 w_k1_FDX_s111
                                 + 0.678514 w_k1_IBM_s111
                                 + 0.830343 w_k1_INTC_s111
                                 + 0.756728 w_k1_KO_s111
                                 + 0.698147 w_k1_MAT_s111
                                 + 0.802855 w_k1_MSFT_s111
                                 + 0.944136 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.10115 w_k1_ADM_s111
                                 + 1.48527 w_k1_AXP_s111
                                 + 1.34077 w_k1_DIS_s111
                                 + 1.19429 w_k1_FDX_s111
                                 + 1.19444 w_k1_IBM_s111
                                 + 1.15663 w_k1_INTC_s111
                                 + 1.24096 w_k1_KO_s111 + 1.47196 w_k1_MAT_s111
                                 + 1.13816 w_k1_MSFT_s111
                                 + 1.22768 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.17046 w_k1_ADM_s111
                                 + 1.17268 w_k1_AXP_s111
                                 + 1.12703 w_k1_DIS_s111
                                 + 1.24086 w_k1_FDX_s111
                                 + 0.807142 w_k1_IBM_s111
                                 + 1.14512 w_k1_INTC_s111
                                 + 0.917691 w_k1_KO_s111
                                 + 0.817746 w_k1_MAT_s111
                                 + 1.149 w_k1_MSFT_s111 + 1.05201 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.11167 w_k1_ADM_s111
                                 + 0.86636 w_k1_AXP_s111
                                 + 1.26381 w_k1_DIS_s111
                                 + 1.02014 w_k1_FDX_s111
                                 + 1.14278 w_k1_IBM_s111
                                 + 0.863366 w_k1_INTC_s111
                                 + 1.1529 w_k1_KO_s111 + 0.922312 w_k1_MAT_s111
                                 + 1.00351 w_k1_MSFT_s111
                                 + 0.989316 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.875888 w_k1_ADM_s111
                                 + 0.868966 w_k1_AXP_s111
                                 + 1.29802 w_k1_DIS_s111
                                 + 1.05284 w_k1_FDX_s111
                                 + 0.749709 w_k1_IBM_s111
                                 + 0.852879 w_k1_INTC_s111
                                 + 0.942451 w_k1_KO_s111
                                 + 0.857459 w_k1_MAT_s111
                                 + 1.06488 w_k1_MSFT_s111
                                 + 1.05026 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.840089 w_k1_ADM_s111
                                 + 0.870289 w_k1_AXP_s111
                                 + 0.867417 w_k1_DIS_s111
                                 + 0.882606 w_k1_FDX_s111
                                 + 1.12256 w_k1_IBM_s111
                                 + 0.792483 w_k1_INTC_s111
                                 + 0.928165 w_k1_KO_s111
                                 + 1.10721 w_k1_MAT_s111
                                 + 0.879598 w_k1_MSFT_s111
                                 + 0.992806 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.885751 w_k1_ADM_s111
                                 + 0.830687 w_k1_AXP_s111
                                 + 0.65052 w_k1_DIS_s111
                                 + 0.79473 w_k1_FDX_s111
                                 + 0.891555 w_k1_IBM_s111
                                 + 0.74371 w_k1_INTC_s111
                                 + 0.737976 w_k1_KO_s111
                                 + 0.913514 w_k1_MAT_s111
                                 + 0.786645 w_k1_MSFT_s111
                                 + 0.85494 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.06995 w_k1_ADM_s111
                                 + 1.33032 w_k1_AXP_s111
                                 + 1.34002 w_k1_DIS_s111
                                 + 1.07164 w_k1_FDX_s111
                                 + 1.55756 w_k1_IBM_s111
                                 + 1.06441 w_k1_INTC_s111
                                 + 1.39204 w_k1_KO_s111 + 1.67245 w_k1_MAT_s111
                                 + 1.05641 w_k1_MSFT_s111
                                 + 1.21501 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.15534 w_k1_ADM_s112
                                 + 0.806627 w_k1_AXP_s112
                                 + 0.860623 w_k1_DIS_s112
                                 + 0.858091 w_k1_FDX_s112
                                 + 1.00202 w_k1_IBM_s112
                                 + 0.897081 w_k1_INTC_s112
                                 + 0.954048 w_k1_KO_s112
                                 + 0.897744 w_k1_MAT_s112
                                 + 0.829986 w_k1_MSFT_s112
                                 + 0.884147 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.17058 w_k1_ADM_s112
                                 + 1.00398 w_k1_AXP_s112
                                 + 1.40727 w_k1_DIS_s112
                                 + 1.10375 w_k1_FDX_s112 + 1.1605 w_k1_IBM_s112
                                 + 0.964519 w_k1_INTC_s112
                                 + 1.19124 w_k1_KO_s112 + 1.35162 w_k1_MAT_s112
                                 + 1.11465 w_k1_MSFT_s112
                                 + 1.02586 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.803614 w_k1_ADM_s112
                                 + 0.640276 w_k1_AXP_s112
                                 + 0.729338 w_k1_DIS_s112
                                 + 0.7611 w_k1_FDX_s112
                                 + 0.669411 w_k1_IBM_s112
                                 + 0.771596 w_k1_INTC_s112
                                 + 0.852018 w_k1_KO_s112
                                 + 0.574869 w_k1_MAT_s112
                                 + 0.850204 w_k1_MSFT_s112
                                 + 0.859107 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.997789 w_k1_ADM_s112
                                 + 0.952047 w_k1_AXP_s112
                                 + 0.999434 w_k1_DIS_s112
                                 + 1.02529 w_k1_FDX_s112
                                 + 0.734714 w_k1_IBM_s112
                                 + 0.971169 w_k1_INTC_s112
                                 + 0.896851 w_k1_KO_s112
                                 + 0.900761 w_k1_MAT_s112
                                 + 0.958863 w_k1_MSFT_s112
                                 + 0.990032 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.13173 w_k1_ADM_s112
                                 + 1.21054 w_k1_AXP_s112
                                 + 1.13683 w_k1_DIS_s112
                                 + 1.00047 w_k1_FDX_s112
                                 + 1.57001 w_k1_IBM_s112
                                 + 1.01835 w_k1_INTC_s112
                                 + 1.35205 w_k1_KO_s112 + 1.41483 w_k1_MAT_s112
                                 + 0.932728 w_k1_MSFT_s112
                                 + 1.00575 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.371 w_k1_ADM_s112
                                 + 1.39101 w_k1_AXP_s112
                                 + 1.41348 w_k1_DIS_s112
                                 + 1.36855 w_k1_FDX_s112
                                 + 1.03424 w_k1_IBM_s112
                                 + 1.04839 w_k1_INTC_s112
                                 + 1.00623 w_k1_KO_s112 + 1.01663 w_k1_MAT_s112
                                 + 1.08229 w_k1_MSFT_s112 + 1.0426 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.927199 w_k1_ADM_s112
                                 + 0.765633 w_k1_AXP_s112
                                 + 0.713854 w_k1_DIS_s112
                                 + 0.934648 w_k1_FDX_s112
                                 + 0.43826 w_k1_IBM_s112
                                 + 0.913483 w_k1_INTC_s112
                                 + 0.639221 w_k1_KO_s112
                                 + 0.608899 w_k1_MAT_s112
                                 + 0.812618 w_k1_MSFT_s112
                                 + 0.823981 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.658619 w_k1_ADM_s112
                                 + 0.790189 w_k1_AXP_s112
                                 + 0.746642 w_k1_DIS_s112
                                 + 0.71007 w_k1_FDX_s112
                                 + 1.19061 w_k1_IBM_s112
                                 + 0.667251 w_k1_INTC_s112
                                 + 0.956855 w_k1_KO_s112
                                 + 1.19682 w_k1_MAT_s112
                                 + 0.718954 w_k1_MSFT_s112
                                 + 0.848346 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.0475 w_k1_ADM_s113
                                 + 1.16955 w_k1_AXP_s113
                                 + 1.18605 w_k1_DIS_s113
                                 + 0.951239 w_k1_FDX_s113
                                 + 1.6199 w_k1_IBM_s113
                                 + 0.941525 w_k1_INTC_s113
                                 + 1.34521 w_k1_KO_s113 + 1.65647 w_k1_MAT_s113
                                 + 0.994651 w_k1_MSFT_s113
                                 + 1.13471 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.84529 w_k1_ADM_s113
                                 + 0.78066 w_k1_AXP_s113
                                 + 0.67142 w_k1_DIS_s113
                                 + 0.855619 w_k1_FDX_s113
                                 + 0.695529 w_k1_IBM_s113
                                 + 0.776544 w_k1_INTC_s113
                                 + 0.779042 w_k1_KO_s113
                                 + 0.723119 w_k1_MAT_s113
                                 + 0.884248 w_k1_MSFT_s113
                                 + 0.961573 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.07022 w_k1_ADM_s113
                                 + 1.05725 w_k1_AXP_s113
                                 + 0.834258 w_k1_DIS_s113
                                 + 0.967333 w_k1_FDX_s113
                                 + 1.05999 w_k1_IBM_s113
                                 + 0.838981 w_k1_INTC_s113
                                 + 0.933278 w_k1_KO_s113
                                 + 1.16009 w_k1_MAT_s113
                                 + 0.984563 w_k1_MSFT_s113
                                 + 1.00232 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.37178 w_k1_ADM_s113
                                 + 1.51473 w_k1_AXP_s113
                                 + 1.11453 w_k1_DIS_s113
                                 + 1.22408 w_k1_FDX_s113
                                 + 0.937008 w_k1_IBM_s113
                                 + 1.19319 w_k1_INTC_s113
                                 + 1.04425 w_k1_KO_s113 + 1.31464 w_k1_MAT_s113
                                 + 1.27456 w_k1_MSFT_s113
                                 + 1.13954 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.27052 w_k1_ADM_s113
                                 + 1.31587 w_k1_AXP_s113
                                 + 1.61661 w_k1_DIS_s113
                                 + 1.35665 w_k1_FDX_s113
                                 + 1.46584 w_k1_IBM_s113
                                 + 1.44054 w_k1_INTC_s113
                                 + 1.46062 w_k1_KO_s113 + 1.40756 w_k1_MAT_s113
                                 + 1.61623 w_k1_MSFT_s113
                                 + 1.38426 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.860546 w_k1_ADM_s113
                                 + 0.784655 w_k1_AXP_s113
                                 + 0.861718 w_k1_DIS_s113
                                 + 0.886743 w_k1_FDX_s113
                                 + 1.07625 w_k1_IBM_s113
                                 + 1.10398 w_k1_INTC_s113
                                 + 1.01879 w_k1_KO_s113
                                 + 0.948071 w_k1_MAT_s113
                                 + 1.06111 w_k1_MSFT_s113
                                 + 1.06017 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.13655 w_k1_ADM_s113
                                 + 1.09956 w_k1_AXP_s113
                                 + 1.28937 w_k1_DIS_s113
                                 + 1.11655 w_k1_FDX_s113 + 1.0803 w_k1_IBM_s113
                                 + 1.12092 w_k1_INTC_s113
                                 + 1.13521 w_k1_KO_s113
                                 + 0.977081 w_k1_MAT_s113
                                 + 1.21706 w_k1_MSFT_s113
                                 + 1.24708 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.40734 w_k1_ADM_s113
                                 + 1.20152 w_k1_AXP_s113
                                 + 1.32902 w_k1_DIS_s113 + 1.3438 w_k1_FDX_s113
                                 + 0.800656 w_k1_IBM_s113
                                 + 1.20591 w_k1_INTC_s113
                                 + 1.11299 w_k1_KO_s113
                                 + 0.837325 w_k1_MAT_s113
                                 + 1.37197 w_k1_MSFT_s113
                                 + 1.18701 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.19153 w_k1_ADM_s114
                                 + 1.51976 w_k1_AXP_s114
                                 + 1.41403 w_k1_DIS_s114
                                 + 1.18178 w_k1_FDX_s114
                                 + 1.65414 w_k1_IBM_s114
                                 + 0.981328 w_k1_INTC_s114
                                 + 1.43171 w_k1_KO_s114 + 1.45659 w_k1_MAT_s114
                                 + 1.02543 w_k1_MSFT_s114
                                 + 1.21903 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.937172 w_k1_ADM_s114
                                 + 0.640898 w_k1_AXP_s114
                                 + 0.654964 w_k1_DIS_s114
                                 + 0.737498 w_k1_FDX_s114
                                 + 0.834724 w_k1_IBM_s114
                                 + 0.722672 w_k1_INTC_s114
                                 + 0.729651 w_k1_KO_s114
                                 + 0.567916 w_k1_MAT_s114
                                 + 0.732741 w_k1_MSFT_s114
                                 + 0.884961 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.948895 w_k1_ADM_s114
                                 + 0.685666 w_k1_AXP_s114
                                 + 0.882515 w_k1_DIS_s114
                                 + 0.923896 w_k1_FDX_s114
                                 + 0.571539 w_k1_IBM_s114
                                 + 0.896905 w_k1_INTC_s114
                                 + 0.729229 w_k1_KO_s114
                                 + 0.387531 w_k1_MAT_s114
                                 + 0.96698 w_k1_MSFT_s114
                                 + 0.864052 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.04258 w_k1_ADM_s114
                                 + 1.09981 w_k1_AXP_s114
                                 + 1.03692 w_k1_DIS_s114
                                 + 1.02206 w_k1_FDX_s114
                                 + 0.978164 w_k1_IBM_s114
                                 + 0.856359 w_k1_INTC_s114
                                 + 0.93677 w_k1_KO_s114
                                 + 0.905602 w_k1_MAT_s114
                                 + 0.996719 w_k1_MSFT_s114
                                 + 1.10596 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0293 w_k1_ADM_s114
                                 + 0.945571 w_k1_AXP_s114
                                 + 0.924692 w_k1_DIS_s114
                                 + 0.912126 w_k1_FDX_s114
                                 + 1.30053 w_k1_IBM_s114
                                 + 0.796062 w_k1_INTC_s114
                                 + 1.12269 w_k1_KO_s114 + 1.43402 w_k1_MAT_s114
                                 + 0.890093 w_k1_MSFT_s114
                                 + 1.13234 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.04511 w_k1_ADM_s114
                                 + 0.684388 w_k1_AXP_s114
                                 + 0.824942 w_k1_DIS_s114
                                 + 0.889826 w_k1_FDX_s114
                                 + 0.930671 w_k1_IBM_s114
                                 + 0.844797 w_k1_INTC_s114
                                 + 0.816845 w_k1_KO_s114
                                 + 0.658523 w_k1_MAT_s114
                                 + 0.913595 w_k1_MSFT_s114
                                 + 0.973711 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.28941 w_k1_ADM_s114
                                 + 1.49196 w_k1_AXP_s114
                                 + 1.49428 w_k1_DIS_s114
                                 + 1.33736 w_k1_FDX_s114
                                 + 0.954956 w_k1_IBM_s114
                                 + 1.24688 w_k1_INTC_s114
                                 + 1.15012 w_k1_KO_s114 + 1.12951 w_k1_MAT_s114
                                 + 1.24794 w_k1_MSFT_s114 + 1.2656 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.03054 w_k1_ADM_s114
                                 + 1.02654 w_k1_AXP_s114
                                 + 0.889741 w_k1_DIS_s114
                                 + 1.08773 w_k1_FDX_s114
                                 + 0.812426 w_k1_IBM_s114
                                 + 1.08476 w_k1_INTC_s114
                                 + 0.813353 w_k1_KO_s114
                                 + 0.537331 w_k1_MAT_s114
                                 + 0.858105 w_k1_MSFT_s114
                                 + 0.913865 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.847341 w_k1_ADM_s115
                                 + 0.881738 w_k1_AXP_s115
                                 + 0.781257 w_k1_DIS_s115
                                 + 0.873923 w_k1_FDX_s115
                                 + 0.619049 w_k1_IBM_s115
                                 + 0.923702 w_k1_INTC_s115
                                 + 0.668089 w_k1_KO_s115
                                 + 0.699612 w_k1_MAT_s115
                                 + 0.847134 w_k1_MSFT_s115
                                 + 0.854373 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.798776 w_k1_ADM_s115
                                 + 0.937743 w_k1_AXP_s115
                                 + 0.831019 w_k1_DIS_s115
                                 + 0.799175 w_k1_FDX_s115
                                 + 1.23583 w_k1_IBM_s115
                                 + 0.897075 w_k1_INTC_s115
                                 + 0.888998 w_k1_KO_s115
                                 + 1.26785 w_k1_MAT_s115
                                 + 0.792937 w_k1_MSFT_s115
                                 + 0.925461 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.11782 w_k1_ADM_s115
                                 + 1.23734 w_k1_AXP_s115
                                 + 1.20858 w_k1_DIS_s115
                                 + 1.07972 w_k1_FDX_s115
                                 + 0.904027 w_k1_IBM_s115
                                 + 1.13267 w_k1_INTC_s115
                                 + 1.08939 w_k1_KO_s115 + 1.1929 w_k1_MAT_s115
                                 + 1.09348 w_k1_MSFT_s115
                                 + 1.02744 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.927489 w_k1_ADM_s115
                                 + 0.904318 w_k1_AXP_s115
                                 + 0.66054 w_k1_DIS_s115
                                 + 0.783628 w_k1_FDX_s115
                                 + 0.762756 w_k1_IBM_s115
                                 + 0.912383 w_k1_INTC_s115
                                 + 0.662995 w_k1_KO_s115
                                 + 0.611323 w_k1_MAT_s115
                                 + 0.766178 w_k1_MSFT_s115
                                 + 0.743009 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.15469 w_k1_ADM_s115
                                 + 1.20128 w_k1_AXP_s115
                                 + 1.47823 w_k1_DIS_s115
                                 + 1.28516 w_k1_FDX_s115
                                 + 0.740995 w_k1_IBM_s115
                                 + 1.29843 w_k1_INTC_s115
                                 + 1.13508 w_k1_KO_s115
                                 + 0.654253 w_k1_MAT_s115
                                 + 1.25962 w_k1_MSFT_s115 + 1.0696 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.14549 w_k1_ADM_s115
                                 + 1.35169 w_k1_AXP_s115
                                 + 1.41539 w_k1_DIS_s115
                                 + 1.13661 w_k1_FDX_s115
                                 + 1.46876 w_k1_IBM_s115
                                 + 1.32491 w_k1_INTC_s115
                                 + 1.31378 w_k1_KO_s115 + 1.62262 w_k1_MAT_s115
                                 + 1.21552 w_k1_MSFT_s115
                                 + 1.22812 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.979678 w_k1_ADM_s115
                                 + 0.907192 w_k1_AXP_s115
                                 + 0.722906 w_k1_DIS_s115
                                 + 0.743357 w_k1_FDX_s115
                                 + 0.840844 w_k1_IBM_s115
                                 + 0.769341 w_k1_INTC_s115
                                 + 0.718422 w_k1_KO_s115
                                 + 0.810667 w_k1_MAT_s115
                                 + 0.686312 w_k1_MSFT_s115
                                 + 0.846248 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.928883 w_k1_ADM_s115
                                 + 0.930812 w_k1_AXP_s115
                                 + 1.16227 w_k1_DIS_s115
                                 + 0.914296 w_k1_FDX_s115
                                 + 1.00974 w_k1_IBM_s115
                                 + 1.20031 w_k1_INTC_s115
                                 + 0.966529 w_k1_KO_s115
                                 + 0.853689 w_k1_MAT_s115
                                 + 1.06196 w_k1_MSFT_s115
                                 + 1.02678 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.22796 w_k1_ADM_s116
                                 + 1.33887 w_k1_AXP_s116
                                 + 1.30239 w_k1_DIS_s116
                                 + 1.29484 w_k1_FDX_s116
                                 + 0.860435 w_k1_IBM_s116
                                 + 1.051 w_k1_INTC_s116 + 0.873388 w_k1_KO_s116
                                 + 1.14188 w_k1_MAT_s116
                                 + 1.11516 w_k1_MSFT_s116
                                 + 1.23429 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.05091 w_k1_ADM_s116
                                 + 1.33265 w_k1_AXP_s116 + 1.3334 w_k1_DIS_s116
                                 + 1.09252 w_k1_FDX_s116
                                 + 1.41975 w_k1_IBM_s116
                                 + 1.07899 w_k1_INTC_s116
                                 + 1.18966 w_k1_KO_s116 + 1.97965 w_k1_MAT_s116
                                 + 1.15257 w_k1_MSFT_s116
                                 + 1.20453 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.04813 w_k1_ADM_s116
                                 + 1.43668 w_k1_AXP_s116
                                 + 0.86057 w_k1_DIS_s116
                                 + 1.05006 w_k1_FDX_s116
                                 + 0.992555 w_k1_IBM_s116
                                 + 1.12837 w_k1_INTC_s116
                                 + 0.885952 w_k1_KO_s116
                                 + 1.31547 w_k1_MAT_s116
                                 + 0.904006 w_k1_MSFT_s116
                                 + 1.06564 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.910511 w_k1_ADM_s116
                                 + 0.90918 w_k1_AXP_s116
                                 + 0.920853 w_k1_DIS_s116
                                 + 1.09512 w_k1_FDX_s116
                                 + 0.823362 w_k1_IBM_s116
                                 + 1.14892 w_k1_INTC_s116
                                 + 0.813489 w_k1_KO_s116
                                 + 0.691906 w_k1_MAT_s116
                                 + 1.05674 w_k1_MSFT_s116
                                 + 0.948406 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.865438 w_k1_ADM_s116
                                 + 0.884346 w_k1_AXP_s116
                                 + 0.684585 w_k1_DIS_s116
                                 + 0.906559 w_k1_FDX_s116
                                 + 0.68761 w_k1_IBM_s116
                                 + 0.792253 w_k1_INTC_s116
                                 + 0.666612 w_k1_KO_s116
                                 + 0.658542 w_k1_MAT_s116
                                 + 0.830145 w_k1_MSFT_s116
                                 + 0.921981 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.766757 w_k1_ADM_s116
                                 + 0.716794 w_k1_AXP_s116
                                 + 0.678008 w_k1_DIS_s116
                                 + 0.710769 w_k1_FDX_s116
                                 + 1.22233 w_k1_IBM_s116
                                 + 0.738407 w_k1_INTC_s116
                                 + 0.707745 w_k1_KO_s116
                                 + 0.869647 w_k1_MAT_s116
                                 + 0.796485 w_k1_MSFT_s116
                                 + 0.943724 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.895783 w_k1_ADM_s116
                                 + 0.670417 w_k1_AXP_s116
                                 + 0.711265 w_k1_DIS_s116
                                 + 0.801214 w_k1_FDX_s116
                                 + 0.935282 w_k1_IBM_s116
                                 + 0.799722 w_k1_INTC_s116
                                 + 0.800481 w_k1_KO_s116
                                 + 0.685867 w_k1_MAT_s116
                                 + 0.873672 w_k1_MSFT_s116
                                 + 0.959315 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.02689 w_k1_ADM_s116
                                 + 0.971957 w_k1_AXP_s116
                                 + 0.981382 w_k1_DIS_s116
                                 + 0.973827 w_k1_FDX_s116
                                 + 1.12705 w_k1_IBM_s116
                                 + 1.00438 w_k1_INTC_s116
                                 + 0.96865 w_k1_KO_s116 + 1.10334 w_k1_MAT_s116
                                 + 0.977037 w_k1_MSFT_s116
                                 + 1.08793 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.36696 w_k1_ADM_s117
                                 + 1.12666 w_k1_AXP_s117
                                 + 0.93351 w_k1_DIS_s117
                                 + 0.969868 w_k1_FDX_s117
                                 + 0.770232 w_k1_IBM_s117
                                 + 0.835954 w_k1_INTC_s117
                                 + 0.824528 w_k1_KO_s117
                                 + 0.674535 w_k1_MAT_s117
                                 + 0.862572 w_k1_MSFT_s117
                                 + 0.867303 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.25555 w_k1_ADM_s117
                                 + 0.969161 w_k1_AXP_s117
                                 + 0.87081 w_k1_DIS_s117
                                 + 0.996038 w_k1_FDX_s117
                                 + 0.645968 w_k1_IBM_s117
                                 + 1.05253 w_k1_INTC_s117
                                 + 0.750035 w_k1_KO_s117
                                 + 0.718173 w_k1_MAT_s117
                                 + 0.975091 w_k1_MSFT_s117
                                 + 0.853085 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.861427 w_k1_ADM_s117
                                 + 0.742055 w_k1_AXP_s117
                                 + 0.68569 w_k1_DIS_s117
                                 + 0.741457 w_k1_FDX_s117
                                 + 0.872558 w_k1_IBM_s117
                                 + 0.75591 w_k1_INTC_s117
                                 + 0.79512 w_k1_KO_s117
                                 + 0.795581 w_k1_MAT_s117
                                 + 0.773266 w_k1_MSFT_s117
                                 + 0.893199 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.13043 w_k1_ADM_s117
                                 + 0.5844 w_k1_AXP_s117
                                 + 0.724154 w_k1_DIS_s117
                                 + 0.706807 w_k1_FDX_s117
                                 + 0.685914 w_k1_IBM_s117
                                 + 0.816541 w_k1_INTC_s117
                                 + 0.760933 w_k1_KO_s117
                                 + 0.515622 w_k1_MAT_s117
                                 + 0.813819 w_k1_MSFT_s117
                                 + 0.791889 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.68734 w_k1_ADM_s117
                                 + 1.5992 w_k1_AXP_s117 + 1.44184 w_k1_DIS_s117
                                 + 1.36063 w_k1_FDX_s117
                                 + 1.00036 w_k1_IBM_s117
                                 + 1.57963 w_k1_INTC_s117
                                 + 1.36491 w_k1_KO_s117
                                 + 0.844994 w_k1_MAT_s117
                                 + 1.45007 w_k1_MSFT_s117
                                 + 1.18405 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.52995 w_k1_ADM_s117
                                 + 1.59111 w_k1_AXP_s117
                                 + 1.83117 w_k1_DIS_s117
                                 + 1.36248 w_k1_FDX_s117
                                 + 1.57642 w_k1_IBM_s117
                                 + 1.65296 w_k1_INTC_s117
                                 + 1.59911 w_k1_KO_s117 + 1.61022 w_k1_MAT_s117
                                 + 1.49614 w_k1_MSFT_s117
                                 + 1.18638 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.15621 w_k1_ADM_s117
                                 + 1.10414 w_k1_AXP_s117
                                 + 1.17111 w_k1_DIS_s117
                                 + 1.04715 w_k1_FDX_s117
                                 + 1.00286 w_k1_IBM_s117
                                 + 1.03809 w_k1_INTC_s117
                                 + 1.14575 w_k1_KO_s117 + 1.03163 w_k1_MAT_s117
                                 + 1.07934 w_k1_MSFT_s117
                                 + 1.05894 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.757527 w_k1_ADM_s117
                                 + 0.933557 w_k1_AXP_s117
                                 + 0.880968 w_k1_DIS_s117
                                 + 0.780664 w_k1_FDX_s117
                                 + 1.25602 w_k1_IBM_s117
                                 + 1.14747 w_k1_INTC_s117
                                 + 1.11962 w_k1_KO_s117 + 1.13445 w_k1_MAT_s117
                                 + 0.941782 w_k1_MSFT_s117
                                 + 0.931019 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00750642557881586 w_k1_ADM_s000
                                 + 0.0119041724645376 w_k1_ADM_s001
                                 + 0.0074678541389898 w_k1_ADM_s002
                                 + 0.00812420065644772 w_k1_ADM_s003
                                 + 0.0154486529162768 w_k1_ADM_s004
                                 + 0.00470256160542928 w_k1_ADM_s005
                                 + 0.0159307798034859 w_k1_ADM_s006
                                 + 0.00859135883838925 w_k1_ADM_s007
                                 + 0.00973315971510881 w_k1_ADM_s010
                                 + 0.00641990116734044 w_k1_ADM_s011
                                 + 0.0140716189682645 w_k1_ADM_s012
                                 + 0.00753664206551924 w_k1_ADM_s013
                                 + 0.015689763010538 w_k1_ADM_s014
                                 + 0.0108632225588218 w_k1_ADM_s015
                                 + 0.0247115203097065 w_k1_ADM_s016
                                 + 0.0135918219833738 w_k1_ADM_s017
                                 + 0.0313855887516603 w_k1_ADM_s100
                                 + 0.00988858295065133 w_k1_ADM_s101
                                 + 0.0192466291232802 w_k1_ADM_s102
                                 + 0.0247132286550601 w_k1_ADM_s103
                                 + 0.0235462728369116 w_k1_ADM_s104
                                 + 0.0166652183876985 w_k1_ADM_s105
                                 + 0.01501485973265 w_k1_ADM_s106
                                 + 0.0314756761035398 w_k1_ADM_s107
                                 + 0.07617986484567 w_k1_ADM_s110
                                 + 0.10635143041271 w_k1_ADM_s111
                                 + 0.0909408317761274 w_k1_ADM_s112
                                 + 0.101942230503952 w_k1_ADM_s113
                                 + 0.0347287023354038 w_k1_ADM_s114
                                 + 0.154146887336444 w_k1_ADM_s115
                                 + 0.0533548176236971 w_k1_ADM_s116
                                 + 0.0483428491715514 w_k1_ADM_s117
                                 + 0.00803359007075415 w_k1_AXP_s000
                                 + 0.0111332739280079 w_k1_AXP_s001
                                 + 0.00770124188293615 w_k1_AXP_s002
                                 + 0.00713164301452405 w_k1_AXP_s003
                                 + 0.0147848269162399 w_k1_AXP_s004
                                 + 0.00517491328380587 w_k1_AXP_s005
                                 + 0.0162767743410632 w_k1_AXP_s006
                                 + 0.00918083336091234 w_k1_AXP_s007
                                 + 0.00891932911514482 w_k1_AXP_s010
                                 + 0.00688470448444999 w_k1_AXP_s011
                                 + 0.0144886301417406 w_k1_AXP_s012
                                 + 0.00591418256947251 w_k1_AXP_s013
                                 + 0.0132626821809559 w_k1_AXP_s014
                                 + 0.0137193387623933 w_k1_AXP_s015
                                 + 0.0246722084260233 w_k1_AXP_s016
                                 + 0.0126739352298048 w_k1_AXP_s017
                                 + 0.0357034393211959 w_k1_AXP_s100
                                 + 0.00840525741185398 w_k1_AXP_s101
                                 + 0.0212357894472869 w_k1_AXP_s102
                                 + 0.0249640686273939 w_k1_AXP_s103
                                 + 0.0266018739578103 w_k1_AXP_s104
                                 + 0.0147831318503654 w_k1_AXP_s105
                                 + 0.016316769436816 w_k1_AXP_s106
                                 + 0.0352904062155419 w_k1_AXP_s107
                                 + 0.0806271621902573 w_k1_AXP_s110
                                 + 0.111077519678885 w_k1_AXP_s111
                                 + 0.086059092702238 w_k1_AXP_s112
                                 + 0.0998306535977529 w_k1_AXP_s113
                                 + 0.0344417792483182 w_k1_AXP_s114
                                 + 0.167635902597748 w_k1_AXP_s115
                                 + 0.0568529304673053 w_k1_AXP_s116
                                 + 0.0449822898345543 w_k1_AXP_s117
                                 + 0.00849827065193904 w_k1_DIS_s000
                                 + 0.0124992575375443 w_k1_DIS_s001
                                 + 0.00800290274488095 w_k1_DIS_s002
                                 + 0.00734532303020216 w_k1_DIS_s003
                                 + 0.0146311656546708 w_k1_DIS_s004
                                 + 0.00491576496775838 w_k1_DIS_s005
                                 + 0.0171087582523133 w_k1_DIS_s006
                                 + 0.0104706771741018 w_k1_DIS_s007
                                 + 0.00909665764225527 w_k1_DIS_s010
                                 + 0.00763743737548653 w_k1_DIS_s011
                                 + 0.0125018899144933 w_k1_DIS_s012
                                 + 0.00716784177662674 w_k1_DIS_s013
                                 + 0.013394020056853 w_k1_DIS_s014
                                 + 0.0123718170658112 w_k1_DIS_s015
                                 + 0.0272149058416002 w_k1_DIS_s016
                                 + 0.0137240870532255 w_k1_DIS_s017
                                 + 0.0374655991105782 w_k1_DIS_s100
                                 + 0.00808309695518581 w_k1_DIS_s101
                                 + 0.0215157651340831 w_k1_DIS_s102
                                 + 0.0248194126010609 w_k1_DIS_s103
                                 + 0.0250746060782222 w_k1_DIS_s104
                                 + 0.0159983215176899 w_k1_DIS_s105
                                 + 0.0141777618672709 w_k1_DIS_s106
                                 + 0.0350461938028055 w_k1_DIS_s107
                                 + 0.0786344845352022 w_k1_DIS_s110
                                 + 0.11437046121145 w_k1_DIS_s111
                                 + 0.0909080590423691 w_k1_DIS_s112
                                 + 0.0986803504087008 w_k1_DIS_s113
                                 + 0.0335612933572224 w_k1_DIS_s114
                                 + 0.170788915874357 w_k1_DIS_s115
                                 + 0.0524668019878243 w_k1_DIS_s116
                                 + 0.0448009795638816 w_k1_DIS_s117
                                 + 0.00770072110635491 w_k1_FDX_s000
                                 + 0.0132551540515207 w_k1_FDX_s001
                                 + 0.00782522740565109 w_k1_FDX_s002
                                 + 0.00710233824376245 w_k1_FDX_s003
                                 + 0.0155510968773241 w_k1_FDX_s004
                                 + 0.00478351217626477 w_k1_FDX_s005
                                 + 0.0156249310966148 w_k1_FDX_s006
                                 + 0.00964244669304824 w_k1_FDX_s007
                                 + 0.00904891796859022 w_k1_FDX_s010
                                 + 0.00693163689677694 w_k1_FDX_s011
                                 + 0.0128491119615016 w_k1_FDX_s012
                                 + 0.00612103510947705 w_k1_FDX_s013
                                 + 0.0132856669609987 w_k1_FDX_s014
                                 + 0.0131628511347128 w_k1_FDX_s015
                                 + 0.0246277914477778 w_k1_FDX_s016
                                 + 0.0142747568801095 w_k1_FDX_s017
                                 + 0.0365983496969701 w_k1_FDX_s100
                                 + 0.00845554950744277 w_k1_FDX_s101
                                 + 0.0202141752588819 w_k1_FDX_s102
                                 + 0.0254467771195946 w_k1_FDX_s103
                                 + 0.0247503482800564 w_k1_FDX_s104
                                 + 0.0145256753165797 w_k1_FDX_s105
                                 + 0.0145016441351061 w_k1_FDX_s106
                                 + 0.0327844379354015 w_k1_FDX_s107
                                 + 0.0784701645029799 w_k1_FDX_s110
                                 + 0.109753357982494 w_k1_FDX_s111
                                 + 0.0861499654632102 w_k1_FDX_s112
                                 + 0.099477544541206 w_k1_FDX_s113
                                 + 0.0334111750899349 w_k1_FDX_s114
                                 + 0.151683140808199 w_k1_FDX_s115
                                 + 0.0538647393874808 w_k1_FDX_s116
                                 + 0.0410009850445602 w_k1_FDX_s117
                                 + 0.00817565137341352 w_k1_IBM_s000
                                 + 0.0134864915427592 w_k1_IBM_s001
                                 + 0.00751328324258053 w_k1_IBM_s002
                                 + 0.00764183141577732 w_k1_IBM_s003
                                 + 0.0159362291313955 w_k1_IBM_s004
                                 + 0.0047449764792956 w_k1_IBM_s005
                                 + 0.0151222412531605 w_k1_IBM_s006
                                 + 0.010381453857102 w_k1_IBM_s007
                                 + 0.0096883626029944 w_k1_IBM_s010
                                 + 0.00705910068141252 w_k1_IBM_s011
                                 + 0.0132526032060571 w_k1_IBM_s012
                                 + 0.00638133911615545 w_k1_IBM_s013
                                 + 0.0137700777552217 w_k1_IBM_s014
                                 + 0.0136504860939672 w_k1_IBM_s015
                                 + 0.0244703969805604 w_k1_IBM_s016
                                 + 0.0148542471819593 w_k1_IBM_s017
                                 + 0.0363411208512622 w_k1_IBM_s100
                                 + 0.0082752703419801 w_k1_IBM_s101
                                 + 0.0212695112619087 w_k1_IBM_s102
                                 + 0.0259079226515559 w_k1_IBM_s103
                                 + 0.0244799636021613 w_k1_IBM_s104
                                 + 0.0146844985535505 w_k1_IBM_s105
                                 + 0.0139778976018066 w_k1_IBM_s106
                                 + 0.0332929015725106 w_k1_IBM_s107
                                 + 0.0806298605328331 w_k1_IBM_s110
                                 + 0.111209775787097 w_k1_IBM_s111
                                 + 0.0873733104136576 w_k1_IBM_s112
                                 + 0.100331520473689 w_k1_IBM_s113
                                 + 0.0339886863543535 w_k1_IBM_s114
                                 + 0.154730187188565 w_k1_IBM_s115
                                 + 0.0540658503165391 w_k1_IBM_s116
                                 + 0.042451620159095 w_k1_IBM_s117
                                 + 0.00841076030120583 w_k1_INTC_s000
                                 + 0.0152198817866084 w_k1_INTC_s001
                                 + 0.0081357404810199 w_k1_INTC_s002
                                 + 0.00694322651211401 w_k1_INTC_s003
                                 + 0.0149173008806359 w_k1_INTC_s004
                                 + 0.00551312524146033 w_k1_INTC_s005
                                 + 0.0147441040172051 w_k1_INTC_s006
                                 + 0.00989984342775158 w_k1_INTC_s007
                                 + 0.00779993374610864 w_k1_INTC_s010
                                 + 0.00564891259850634 w_k1_INTC_s011
                                 + 0.0126323062548742 w_k1_INTC_s012
                                 + 0.00547134900116456 w_k1_INTC_s013
                                 + 0.0136774816300936 w_k1_INTC_s014
                                 + 0.012436207503939 w_k1_INTC_s015
                                 + 0.0246788142513007 w_k1_INTC_s016
                                 + 0.0148021199698842 w_k1_INTC_s017
                                 + 0.0394436863412856 w_k1_INTC_s100
                                 + 0.00948831681315301 w_k1_INTC_s101
                                 + 0.0185671865544747 w_k1_INTC_s102
                                 + 0.0244661496165497 w_k1_INTC_s103
                                 + 0.0248870918651266 w_k1_INTC_s104
                                 + 0.013140344510673 w_k1_INTC_s105
                                 + 0.0142876821498683 w_k1_INTC_s106
                                 + 0.0336949590614198 w_k1_INTC_s107
                                 + 0.0810945019911343 w_k1_INTC_s110
                                 + 0.102720494316106 w_k1_INTC_s111
                                 + 0.0786614451184965 w_k1_INTC_s112
                                 + 0.099333504770684 w_k1_INTC_s113
                                 + 0.0304713668299705 w_k1_INTC_s114
                                 + 0.168768460847799 w_k1_INTC_s115
                                 + 0.0517686196665088 w_k1_INTC_s116
                                 + 0.0468155165460742 w_k1_INTC_s117
                                 + 0.00747233210110488 w_k1_KO_s000
                                 + 0.0142613162042167 w_k1_KO_s001
                                 + 0.00736525385523829 w_k1_KO_s002
                                 + 0.00696531968816696 w_k1_KO_s003
                                 + 0.0162285559234444 w_k1_KO_s004
                                 + 0.00472046065003666 w_k1_KO_s005
                                 + 0.0148684359390475 w_k1_KO_s006
                                 + 0.010158600580373 w_k1_KO_s007
                                 + 0.00897225375310547 w_k1_KO_s010
                                 + 0.00657788060415063 w_k1_KO_s011
                                 + 0.0134736690791119 w_k1_KO_s012
                                 + 0.00629238436007494 w_k1_KO_s013
                                 + 0.0132186983632076 w_k1_KO_s014
                                 + 0.0121988064975954 w_k1_KO_s015
                                 + 0.0249970479893328 w_k1_KO_s016
                                 + 0.0148409890744255 w_k1_KO_s017
                                 + 0.0357481940665541 w_k1_KO_s100
                                 + 0.00865007975813281 w_k1_KO_s101
                                 + 0.0202166771015942 w_k1_KO_s102
                                 + 0.024069291289685 w_k1_KO_s103
                                 + 0.0275441969147472 w_k1_KO_s104
                                 + 0.0147739116347087 w_k1_KO_s105
                                 + 0.0135802788606655 w_k1_KO_s106
                                 + 0.0290784864957983 w_k1_KO_s107
                                 + 0.0713741738800532 w_k1_KO_s110
                                 + 0.108559615464651 w_k1_KO_s111
                                 + 0.086659461613969 w_k1_KO_s112
                                 + 0.10096752312107 w_k1_KO_s113
                                 + 0.0322509456376991 w_k1_KO_s114
                                 + 0.153791904919957 w_k1_KO_s115
                                 + 0.0476485903743393 w_k1_KO_s116
                                 + 0.0450769850156245 w_k1_KO_s117
                                 + 0.010811844623336 w_k1_MAT_s000
                                 + 0.014444844776894 w_k1_MAT_s001
                                 + 0.00770221004201679 w_k1_MAT_s002
                                 + 0.00631858245249138 w_k1_MAT_s003
                                 + 0.0185754976669155 w_k1_MAT_s004
                                 + 0.0050685910855495 w_k1_MAT_s005
                                 + 0.0170256180207489 w_k1_MAT_s006
                                 + 0.0109864348370308 w_k1_MAT_s007
                                 + 0.00911801183310759 w_k1_MAT_s010
                                 + 0.00677655684310818 w_k1_MAT_s011
                                 + 0.0128055457012948 w_k1_MAT_s012
                                 + 0.00643722084411275 w_k1_MAT_s013
                                 + 0.0145314628007898 w_k1_MAT_s014
                                 + 0.0145522072882397 w_k1_MAT_s015
                                 + 0.0220283062548335 w_k1_MAT_s016
                                 + 0.0160566757391887 w_k1_MAT_s017
                                 + 0.0379152910879036 w_k1_MAT_s100
                                 + 0.00736719325771956 w_k1_MAT_s101
                                 + 0.0297631128886764 w_k1_MAT_s102
                                 + 0.0262378262560309 w_k1_MAT_s103
                                 + 0.0272596614504115 w_k1_MAT_s104
                                 + 0.0171481798756223 w_k1_MAT_s105
                                 + 0.0161908347234512 w_k1_MAT_s106
                                 + 0.0343035897419028 w_k1_MAT_s107
                                 + 0.0783961605198987 w_k1_MAT_s110
                                 + 0.117071228939608 w_k1_MAT_s111
                                 + 0.0872608919991855 w_k1_MAT_s112
                                 + 0.101995559891908 w_k1_MAT_s113
                                 + 0.0302719121924654 w_k1_MAT_s114
                                 + 0.164446290804409 w_k1_MAT_s115
                                 + 0.0601327881500522 w_k1_MAT_s116
                                 + 0.0399309201125675 w_k1_MAT_s117
                                 + 0.00816035615941275 w_k1_MSFT_s000
                                 + 0.0133459258278319 w_k1_MSFT_s001
                                 + 0.00796208871279561 w_k1_MSFT_s002
                                 + 0.0067627167743194 w_k1_MSFT_s003
                                 + 0.0155923660803211 w_k1_MSFT_s004
                                 + 0.00455494728476204 w_k1_MSFT_s005
                                 + 0.0152505758401375 w_k1_MSFT_s006
                                 + 0.0102065351597452 w_k1_MSFT_s007
                                 + 0.00811524568190375 w_k1_MSFT_s010
                                 + 0.00653097736306933 w_k1_MSFT_s011
                                 + 0.0114406546729396 w_k1_MSFT_s012
                                 + 0.00654618215520769 w_k1_MSFT_s013
                                 + 0.0123922330787485 w_k1_MSFT_s014
                                 + 0.0118279748216924 w_k1_MSFT_s015
                                 + 0.0248011020201826 w_k1_MSFT_s016
                                 + 0.0141281902768779 w_k1_MSFT_s017
                                 + 0.0379412442963301 w_k1_MSFT_s100
                                 + 0.00852722660330655 w_k1_MSFT_s101
                                 + 0.0193705188673318 w_k1_MSFT_s102
                                 + 0.0241002304182402 w_k1_MSFT_s103
                                 + 0.0251038106183116 w_k1_MSFT_s104
                                 + 0.0134512795479649 w_k1_MSFT_s105
                                 + 0.0142198429537836 w_k1_MSFT_s106
                                 + 0.0318862355238422 w_k1_MSFT_s107
                                 + 0.0774218168137073 w_k1_MSFT_s110
                                 + 0.106528335095651 w_k1_MSFT_s111
                                 + 0.0796689525573062 w_k1_MSFT_s112
                                 + 0.107929653134748 w_k1_MSFT_s113
                                 + 0.0308589989416409 w_k1_MSFT_s114
                                 + 0.15572302659068 w_k1_MSFT_s115
                                 + 0.0522765698806614 w_k1_MSFT_s116
                                 + 0.0439384890578305 w_k1_MSFT_s117
                                 + 0.00769700060212844 w_k1_PG_s000
                                 + 0.0127367480902107 w_k1_PG_s001
                                 + 0.00670724108606732 w_k1_PG_s002
                                 + 0.00746555608601402 w_k1_PG_s003
                                 + 0.0167561527990148 w_k1_PG_s004
                                 + 0.00458236252096563 w_k1_PG_s005
                                 + 0.0161673167314899 w_k1_PG_s006
                                 + 0.00969814290515151 w_k1_PG_s007
                                 + 0.00929703769488856 w_k1_PG_s010
                                 + 0.00604436246714664 w_k1_PG_s011
                                 + 0.0118639847807342 w_k1_PG_s012
                                 + 0.00618422120566611 w_k1_PG_s013
                                 + 0.0126438176257401 w_k1_PG_s014
                                 + 0.011853338249853 w_k1_PG_s015
                                 + 0.0269188258800344 w_k1_PG_s016
                                 + 0.0134108850083893 w_k1_PG_s017
                                 + 0.0393285163832909 w_k1_PG_s100
                                 + 0.00794883134640152 w_k1_PG_s101
                                 + 0.021705897155921 w_k1_PG_s102
                                 + 0.0250886699049581 w_k1_PG_s103
                                 + 0.0266574783485047 w_k1_PG_s104
                                 + 0.0144540455784786 w_k1_PG_s105
                                 + 0.0145845248438332 w_k1_PG_s106
                                 + 0.0316170623122898 w_k1_PG_s107
                                 + 0.0766405718640537 w_k1_PG_s110
                                 + 0.114714743199163 w_k1_PG_s111
                                 + 0.0808469260168213 w_k1_PG_s112
                                 + 0.106655640628096 w_k1_PG_s113
                                 + 0.0343329204473566 w_k1_PG_s114
                                 + 0.154000290872953 w_k1_PG_s115
                                 + 0.0569040188748741 w_k1_PG_s116
                                 + 0.0408998744332336 w_k1_PG_s117 >= 1.17
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
