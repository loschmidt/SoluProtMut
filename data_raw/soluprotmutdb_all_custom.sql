-- plus following changes in the outpus: PDB for PKS -> pks_itasser, PDB for TEM -> 4GKU
SELECT `variants`.`sequence_id`          AS `sequence_id`,
       `results`.`var_id`                AS `var_id`,
       `results`.`setup_id`              AS `setup_id`,
       `results`.`result_id`             AS `result_id`,
       `results`.`curated`               AS `curated`,
       `results`.`pre_counts`            AS `pre_counts`,
       `results`.`post_counts`           AS `post_counts`,
       `results`.`raw_score`             AS `raw_score`,
       `results`.`processed_score`       AS `processed_score`,
       `results`.`total_concentration`   AS `total_concentration`,
       `results`.`soluble_concentration` AS `soluble_concentration`,
       `results`.`fraction_in_total`     AS `fraction_in_total`,
       `results`.`fraction_in_soluble`   AS `fraction_in_soluble`,
       `results`.`solubility`            AS `solubility`,
       `results`.`solubility_change`     AS `solubility_change`,
       `results`.`solubility_effect`     AS `solubility_effect`,
       `d`.`dataset_names`                             AS `dataset_names`,
       `spa`.`doi`                                     AS `doi`,
       `spa`.`publication_id`                          AS `publication_id`,
       `spa`.`host_cell`                               AS `host_cell`,
       `spa`.`host_strain`                             AS `host_strain`,
       `spa`.`system_type`                             AS `system_type`,
       `spa`.`t`                                       AS `T`,
       `spa`.`ph`                                      AS `pH`,
       `spa`.`levels`                                  AS `levels`,
       `spa`.`notes`                                   AS `notes`,
       `spa`.`assay_type`                              AS `assay_type`,
       `spa`.`assay_name`                              AS `assay_name`,
       `bu`.`interpro_entry_ids`                       AS `interpro_entry_ids`,
       `bu`.`pdb_ids`                                  AS `pdb_id`,
       `var`.`positions`                               AS `position`,
       `var`.`pdb_positions`                           AS `pdb_position`,
       `var`.`wild_types`                              AS `wild_type`,
       `var`.`substituents`                            AS `mutation`,
       `var`.`mutations`                               AS `mutations`,
       `var`.`reverse_mutations`                       AS `reverse_mutations`,
       `var`.`chains`                                  AS `chain`,
       IF(`var`.`multiplepoint` <= 1, 1, 0)             AS `is_singlepoint`,
       `protein_sequence`.`uniprot_id`   AS `uniprot_id`,
       `protein_sequence`.`protein_name` AS `protein_name`,
       `protein_sequence`.`species`      AS `species`,
       `protein_sequence`.`ec_number`    AS `ec_number`,
       `protein_sequence`.`protein_seq`  AS `protein_seq`
FROM   ((((((`results`
             LEFT JOIN `variants`
                    ON( `results`.`var_id` = `variants`.`var_id` ))
            LEFT JOIN (SELECT `datapoints_in_dataset`.`result_id`           AS `result_id`,
                              Group_concat(`datasets`.`name` SEPARATOR '|') AS `dataset_names`
                       FROM   (`datapoints_in_dataset`
                               JOIN `datasets`
                                 ON( `datapoints_in_dataset`.`dataset_id` = `datasets`.`dataset_id` ))
                       GROUP  BY `datapoints_in_dataset`.`result_id`) `d`
                   ON( `results`.`result_id` = `d`.`result_id` ))
           LEFT JOIN (SELECT `s`.`setup_id`       AS `setup_id`,
                             `p`.`doi`            AS `doi`,
                             `p`.`publication_id` AS `publication_id`,
                             `s`.`host_cell`      AS `host_cell`,
                             `s`.`host_strain`    AS `host_strain`,
                             `s`.`system_type`    AS `system_type`,
                             `s`.`t`              AS `T`,
                             `s`.`ph`             AS `pH`,
                             `s`.`levels`         AS `levels`,
                             `s`.`notes`          AS `notes`,
                             `s`.`assay_type`     AS `assay_type`,
                             `a`.`name`           AS `assay_name`
                      FROM   ((`setups` `s`
                               LEFT JOIN `assays` `a`
                                      ON( `s`.`assay_type` = `a`.`assay_id` ))
                              LEFT JOIN (SELECT `publications`.`doi`            AS `doi`,
                                                `publications`.`publication_id` AS `publication_id`
                                         FROM   `publications`) `p`
                                     ON( `s`.`publication_id` = `p`.`publication_id` ))) `spa`
                  ON( `results`.`setup_id` = `spa`.`setup_id` ))
          LEFT JOIN (SELECT `protein_sequence_interpro_entries`.`sequence_id`                                   AS `sequence_id`,
                            Group_concat(`protein_sequence_interpro_entries`.`interpro_entry_id` SEPARATOR '|') AS `interpro_entry_ids`,
                            `psbu`.`pdb_ids`                                                                                  AS `pdb_ids`
                     FROM   (`protein_sequence_interpro_entries`
                             LEFT JOIN (SELECT `protein_sequence_biological_unit`.`sequence_id`       AS `sequence_id`,
                                               Group_concat(`biological_unit`.`pdb_id` SEPARATOR '|') AS `pdb_ids`
                                        FROM   (`protein_sequence_biological_unit`
                                                JOIN `biological_unit`
                                                  ON( `protein_sequence_biological_unit`.`bio_unit_id` = `biological_unit`.`bio_unit_id` ))
                                        GROUP  BY `protein_sequence_biological_unit`.`sequence_id`) `psbu`
                                    ON( `protein_sequence_interpro_entries`.`sequence_id` = `psbu`.`sequence_id` ))
                     GROUP  BY `protein_sequence_interpro_entries`.`sequence_id`) `bu`
                 ON( `variants`.`sequence_id` = `bu`.`sequence_id` ))
         LEFT JOIN (SELECT `mut`.`var_id`                                                AS `var_id`,
                           Group_concat(`mut`.`wild_type` SEPARATOR ',')                 AS `wild_types`,
                           Group_concat(`mut`.`substituent` SEPARATOR ',')               AS `substituents`,
                           Group_concat(`mut`.`position` SEPARATOR ',')                  AS `positions`,
                           Group_concat(`mut`.`structure_index` SEPARATOR ',')           AS `pdb_positions`,
                           Group_concat(DISTINCT `mut`.`mutation` SEPARATOR ',')         AS `mutations`,
                           Group_concat(DISTINCT `mut`.`reverse_mutation` SEPARATOR ',') AS `reverse_mutations`,
                           Group_concat(DISTINCT `mut`.`chain` SEPARATOR ',')            AS `chains`,
                           Count(0)                                                      AS `multiplepoint`
                    FROM   (SELECT `mutations`.`var_id`                                                                    AS `var_id`,
								   `mutations`.`substituent` AS substituent,
								   `ps`.`residue` AS wild_type,
								   `mutations`.`position` AS position,
                                   Concat(`ps`.`residue`, `mutations`.`position`, `mutations`.`substituent`) AS `mutation`,
                                   Concat(`mutations`.`substituent`, `mutations`.`position`, `ps`.`residue`) AS `reverse_mutation`,
								   `sr`.`structure_index` AS structure_index,
                                   `sr`.`chain` AS `chain`
                            FROM   ((`mutations`
                                     JOIN `protein_residues` `ps`
                                       ON( `mutations`.`position` = `ps`.`position`
                                           AND `mutations`.`sequence_id` = `ps`.`sequence_id` ))
                                    LEFT JOIN (SELECT `structure_residues`.`new_chain`   AS `chain`,
                                                      `structure_residues`.`sequence_id` AS `sequence_id`,
                                                      `structure_residues`.`position`    AS `position`,
													  `structure_residues`.`structure_index` AS `structure_index`
                                               FROM   `structure_residues`) `sr`
                                           ON( `mutations`.`position` = `sr`.`position`
                                               AND `mutations`.`sequence_id` = `sr`.`sequence_id` ))) `mut`
                    GROUP  BY `mut`.`var_id`) `var`
                ON( `results`.`var_id` = `var`.`var_id` ))
        LEFT JOIN `protein_sequence`
               ON( `variants`.`sequence_id` = `protein_sequence`.`sequence_id` )) 