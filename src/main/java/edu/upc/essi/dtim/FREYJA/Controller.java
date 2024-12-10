package edu.upc.essi.dtim.FREYJA;

import org.json.simple.JSONArray;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.bind.annotation.RestController;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import static edu.upc.essi.dtim.FREYJA.Main.*;

@RestController
public class Controller {

    @PostMapping("/profile")
    public ResponseEntity<?> createProfileAPI(@RequestParam("filePath") String filePath,
                                              @RequestParam("storePath") String storePath) {
        try {
            return ResponseEntity.ok(createProfile(filePath, storePath));
        }
        catch (Exception e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/profilesOfFolder")
    public ResponseEntity<Boolean> profilesOfFolder(@RequestParam("directoryPath") String directoryPath,
                                                    @RequestParam("storePath") String storePath) {
        String scriptPath = "C:\\Projects\\FREYJA\\generate_profiles.ps1"; // path to the script

        // Construct the PowerShell command
        String command = String.format("powershell.exe -ExecutionPolicy Bypass -File \"%s\" -directoryBenchmark  \"%s\" -directoryStoreProfiles \"%s\"",
                scriptPath, directoryPath, storePath);

        try {
            ProcessBuilder pb = new ProcessBuilder("cmd.exe", "/c", command); // Use ProcessBuilder to execute the command
            pb.redirectErrorStream(true); // Merge stdout and stderr
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) { // Read the output
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }
            }

            int exitCode = process.waitFor(); // Wait for the process to complete and get the exit code
            System.out.println("Exit Code: " + exitCode);

        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PostMapping("/computeDistances")
    public ResponseEntity<?> computeDistancesAPI(@RequestParam("queryDataset") String queryDataset,
                                                    @RequestParam("queryColumn") String queryColumn,
                                                    @RequestParam("profilesFolder") String profilesFolder,
                                                    @RequestParam("distancesFolder") String distancesFolder) {
        try {
            return ResponseEntity.ok(computeDistances(queryDataset, queryColumn, profilesFolder, distancesFolder));
        }
        catch (Exception e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/computeDistancesTwoFiles")
    public ResponseEntity<?> calculateDistancesTwoFilesAPI(@RequestParam("csvFilePath1") String csvFilePath1,
                                                             @RequestParam("csvFilePath2") String csvFilePath2,
                                                             @RequestParam("pathToWriteDistances") String pathToWriteDistances) {
        try {
            return ResponseEntity.ok(calculateDistancesTwoFiles(csvFilePath1, csvFilePath2, pathToWriteDistances));
        }
        catch (Exception e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
